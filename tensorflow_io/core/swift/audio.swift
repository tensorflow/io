import AVFoundation

@_silgen_name("DecodeAACFunctionCall")
func DecodeAACFunctionCall(state: UnsafeMutableRawPointer, codec: Int64, rate: Int64, channels: Int64, data_in_chunk: UnsafeRawPointer, size_in_chunk: UnsafePointer<Int64>, chunk: Int64, frames: Int64, data_out: UnsafeMutableRawPointer, size_out: Int64) -> Int {
    
    let header_bytes = Int64(7)
    var stream_description_in = AudioStreamBasicDescription(
        mSampleRate: Double(rate),
        mFormatID: kAudioFormatMPEG4AAC,
        mFormatFlags: 0,
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
        let data_in = data_in_chunk.advanced(by: Int(offset))
        let size_in = size_in_chunk.advanced(by: i).pointee
        let buffer_in_data_offset = Int64(offset - Int64(i) * header_bytes)
        buffer_in.data.advanced(by: Int(buffer_in_data_offset)).copyMemory(from:data_in.advanced(by: Int(header_bytes)), byteCount:Int(size_in - header_bytes))
        buffer_in.packetDescriptions!.advanced(by: i).pointee = AudioStreamPacketDescription(mStartOffset: buffer_in_data_offset, mVariableFramesInPacket: UInt32(0), mDataByteSize: UInt32(size_in - header_bytes))
        
        offset += Int64(size_in)
    }
    
    buffer_in.byteLength = UInt32(offset - chunk * header_bytes)
    buffer_in.packetCount = UInt32(chunk)
    
    guard let format_out = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(rate),
        channels: UInt32(channels),
        interleaved: true) else {
            return -2
    }
    
    let frameCapacity = UInt32(frames)
    
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
        return -5
    }
    if error != nil {
        return -6
    }
    
    let data_out_source = buffer_out.floatChannelData!.pointee
    let size_out_source = Int(buffer_out.frameLength) * Int(channels) * MemoryLayout<Float>.size
    
    
    data_out.copyMemory(from: data_out_source, byteCount: size_out_source)
    
    return 0
}

typealias EncodeAACContext = (rate: Int64, channels: Int64, buffer_out: AVAudioCompressedBuffer)

@_silgen_name("EncodeAACFunctionFini")
func EncodeAVCFunctionFini(context: UnsafeMutablePointer<EncodeAACContext>) -> Void {
    context.pointee.buffer_out = AVAudioCompressedBuffer()
    context.deinitialize(count: 1)
    context.deallocate()
}

@_silgen_name("EncodeAACFunctionInit")
func EncodeAACFunctionInit(codec: Int64, rate: Int64, channels: Int64) -> UnsafeMutablePointer<EncodeAACContext>?  {
    
    let context = UnsafeMutablePointer<EncodeAACContext>.allocate(capacity: 1)
    context.initialize(to: (rate: rate, channels: channels, buffer_out: AVAudioCompressedBuffer()))
    
    return context
}

@_silgen_name("EncodeAACFunctionCall")
func EncodeAACFunctionCall(context: UnsafeMutablePointer<EncodeAACContext>, data_in: UnsafePointer<Float>, size_in: Int64, data_out_chunk: UnsafeMutablePointer<UnsafeMutableRawPointer>, size_out_chunk: UnsafeMutablePointer<Int64>, chunk: UnsafeMutablePointer<Int64>) -> Int64 {
    
    let rate = context.pointee.rate
    let channels = context.pointee.channels
    let frames = size_in / channels
    let packetCapacity = frames / 1024
    
    let format_in = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(rate),
        channels: UInt32(channels),
        interleaved: true)
    
    let frameCapacity = UInt32(frames)
    let buffer_in = AVAudioPCMBuffer(pcmFormat:format_in!, frameCapacity: frameCapacity)
    buffer_in!.frameLength = frameCapacity
    
    buffer_in!.floatChannelData!.pointee.initialize(from: data_in, count: Int(size_in))
    
    var stream_description_out = AudioStreamBasicDescription(
        mSampleRate: Double(rate),
        mFormatID: kAudioFormatMPEG4AAC,
        mFormatFlags: 0,
        mBytesPerPacket: 0,
        mFramesPerPacket:  0,
        mBytesPerFrame: 0,
        mChannelsPerFrame: UInt32(channels),
        mBitsPerChannel: 0,
        mReserved: 0)
    
    let format_out = AVAudioFormat(streamDescription: &stream_description_out)
    
    let maximumPacketSize = 4096
    
    let buffer_out = AVAudioCompressedBuffer(
        format: format_out!,
        packetCapacity: UInt32(packetCapacity),
        maximumPacketSize: maximumPacketSize)
    
    let converter = AVAudioConverter(from: format_in!, to: format_out!)
    //converter!.bitRate = onverter!.applicableEncodeBitRates[0]
    
    var haveData = true
    let input_block : AVAudioConverterInputBlock = { packetCount, inputStatus in
        if (!haveData) {
            inputStatus.pointee = AVAudioConverterInputStatus.endOfStream;
            return nil
        } else {
            haveData = false
            inputStatus.pointee = AVAudioConverterInputStatus.haveData
            return  buffer_in
        }
    }
    var error : NSError?
    let status = converter!.convert(to: buffer_out, error: &error, withInputFrom: input_block)
    switch (status)
    {
    case .haveData:
        break
    default:
        return -1
    }
    if error != nil {
        return -2
    }
    
    if chunk.pointee < buffer_out.packetCount {
        return -3
    }
    
    for i in 0..<buffer_out.packetCount {
        let packetDescription = buffer_out.packetDescriptions!.advanced(by: Int(i)).pointee
        let data_p = buffer_out.data.advanced(by: Int(packetDescription.mStartOffset))
        let size_p = packetDescription.mDataByteSize
        data_out_chunk.advanced(by: Int(i)).pointee = data_p
        size_out_chunk.advanced(by: Int(i)).pointee = Int64(size_p)
    }
    chunk.pointee = Int64(buffer_out.packetCount)
    
    context.pointee.buffer_out = buffer_out
    
    return 0
}
