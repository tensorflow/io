import AVFoundation

@_silgen_name("DecodeAACFunctionCall")
func DecodeAACFunctionCall(state: UnsafeMutableRawPointer, codec: Int64, rate: Int64, channels: Int64, frame_in_chunk: UnsafePointer<Int64>, data_in_chunk: UnsafePointer<UnsafeRawPointer>, size_in_chunk: UnsafePointer<Int64>, chunk: Int64, data_out: UnsafeMutableRawPointer, size_out: Int64) -> Int {
    
    let header_bytes = Int64(7)
    var stream_description_in = AudioStreamBasicDescription(
        mSampleRate: Double(rate),
        mFormatID: kAudioFormatMPEG4AAC,
        mFormatFlags: UInt32(MPEG4ObjectID.AAC_LC.rawValue),
        mBytesPerPacket: 0,
        mFramesPerPacket:  1024,
        mBytesPerFrame: 0,
        mChannelsPerFrame: UInt32(channels),
        mBitsPerChannel: 0,
        mReserved: 0)
    guard let format_in = AVAudioFormat(streamDescription: &stream_description_in) else {
        return -1
    }
    
    var maximumPacketSize = 0
    for i in 0..<Int(chunk) {
        let size_in = size_in_chunk.advanced(by: i).pointee
        if maximumPacketSize < Int(size_in - header_bytes) {
            maximumPacketSize = Int(size_in - header_bytes)
        }
    }
    
    let buffer_in = AVAudioCompressedBuffer(
        format: format_in,
        packetCapacity: UInt32(chunk),
        maximumPacketSize: maximumPacketSize)
    
    var offset: Int64 = 0
    for i in 0..<Int(chunk) {
        let data_in = data_in_chunk.advanced(by: i).pointee
        let size_in = size_in_chunk.advanced(by: i).pointee
        buffer_in.data.advanced(by: Int(offset)).copyMemory(from:data_in.advanced(by: Int(header_bytes)), byteCount:Int(size_in - header_bytes))
        buffer_in.packetDescriptions!.advanced(by: i).pointee = AudioStreamPacketDescription(mStartOffset: offset, mVariableFramesInPacket: UInt32(0), mDataByteSize: UInt32(size_in - header_bytes))
        
        offset += Int64(size_in - header_bytes)
    }
    
    buffer_in.byteLength = UInt32(offset)
    buffer_in.packetCount = UInt32(chunk)
    
    guard let format_out = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(rate),
        channels: UInt32(channels),
        interleaved: true) else {
            return -2
    }
    
    var frameCapacity: UInt32 = 0
    for i in 0..<Int(chunk) {
        frameCapacity += UInt32(frame_in_chunk.advanced(by: i).pointee)
    }
    guard let buffer_out = AVAudioPCMBuffer(pcmFormat:format_out, frameCapacity: frameCapacity) else {
        return -3
    }
    
    guard let converter = AVAudioConverter(from: format_in, to: format_out) else {
        return -4
    }
    
    converter.primeMethod = AVAudioConverterPrimeMethod.none
    converter.primeInfo = AVAudioConverterPrimeInfo.init(leadingFrames: 0, trailingFrames: 0)
    
    var haveData = true
    let input_block : AVAudioConverterInputBlock = { packetCount, inputStatus in
        if (!haveData) {
            inputStatus.pointee = AVAudioConverterInputStatus.endOfStream;
            return nil;
        } else {
            haveData = false
            inputStatus.pointee = AVAudioConverterInputStatus.haveData
            return  buffer_in; // fill and return input buffer
        }
    }
    
    var error : NSError?
    let status = converter.convert(to: buffer_out, error: &error, withInputFrom: input_block)
    switch (status)
    {
    case .haveData:
        break
    default:
        return -6
    }
    if error != nil {
        return -7
    }
    
    let data_out_source = buffer_out.floatChannelData!.pointee
    let size_out_source = Int(buffer_out.frameLength) * Int(channels) * MemoryLayout<Float>.size
    
    
    data_out.copyMemory(from: data_out_source, byteCount: size_out_source)
    
    return 0
}
