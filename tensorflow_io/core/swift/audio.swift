import AVFoundation

@_silgen_name("DecodeAACFunctionCall")
func DecodeAACFunctionCall(state: UnsafeMutableRawPointer, codec: Int64, rate: Int64, channels: Int64, frames: Int64, data_in: UnsafeRawPointer, size_in: Int64, data_out: UnsafeMutableRawPointer, size_out: Int64) -> Int {
    
    let header_bytes = Int64(7)
    var stream_description_in = AudioStreamBasicDescription(
        mSampleRate: Double(rate),
        mFormatID: kAudioFormatMPEG4AAC,
        mFormatFlags: 0,
        mBytesPerPacket: 0,
        mFramesPerPacket:  0,
        mBytesPerFrame: 0,
        mChannelsPerFrame: UInt32(channels),
        mBitsPerChannel: 0,
        mReserved: 0)
    guard let format_in = AVAudioFormat(streamDescription: &stream_description_in) else {
        return -1
    }
    
    let buffer_in = AVAudioCompressedBuffer(
        format: format_in,
        packetCapacity: 1,
        maximumPacketSize: Int(size_in - header_bytes))
    
    buffer_in.data.copyMemory(from:data_in.advanced(by: Int(header_bytes)), byteCount:Int(size_in - header_bytes))
    buffer_in.byteLength = UInt32(size_in - header_bytes)
    buffer_in.packetCount = 1
    buffer_in.packetDescriptions!.pointee.mDataByteSize = UInt32(size_in - header_bytes)
    buffer_in.packetDescriptions!.pointee.mStartOffset = 0
    buffer_in.packetDescriptions!.pointee.mVariableFramesInPacket = 0
    
    guard let format_out = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(rate),
        channels: UInt32(channels),
        interleaved: true) else {
            return -2
    }
    
    guard let buffer_out = AVAudioPCMBuffer(pcmFormat:format_out, frameCapacity: UInt32(frames)) else {
        return -3
    }
    
    guard let converter = AVAudioConverter(from: format_in, to: format_out) else {
        return -4
    }
    
    let input_block : AVAudioConverterInputBlock = {
        packetCount, inputStatus in inputStatus.pointee = AVAudioConverterInputStatus.haveData
        return buffer_in
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
    let size_out_source = Int(buffer_out.frameLength) * Int(channels) * 4 // sizeof(float) = 4
    
    if (size_out_source != size_out) {
        return -8
    }
    data_out.copyMemory(from: data_out_source, byteCount: size_out_source)
    
    return 0
}
