import AVFoundation
import VideoToolbox

class VideoDataOutputSampleBufferDelegate : NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var bytes: Int64
    var width: Int64
    var height: Int64
    var copied: Int64
    var buffer: UnsafeMutableRawPointer?
    var semaphore_in:  DispatchSemaphore
    var semaphore_out:  DispatchSemaphore
    
    init(semaphore_in: DispatchSemaphore, semaphore_out: DispatchSemaphore) {
        self.bytes = 0
        self.width = 0
        self.height = 0
        self.copied = 0
        self.buffer = nil
        self.semaphore_in = semaphore_in
        self.semaphore_out = semaphore_out
        super.init()
    }
    
    deinit {
        // TODO: This is not invoked, memory leak?
        print("VideoDataOutputSampleBufferDelegate.deinit")
    }
    
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        print("frame dropped: \(sampleBuffer)")
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        semaphore_in.wait()
        
        defer { semaphore_out.signal() }
        
        if sampleBuffer.numSamples != 1 {
            print("number of samples \(sampleBuffer.numSamples) is not supported")
            return
        }
        
        let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer!)
        let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer!)
        
        if pixelFormat != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || planeCount != 2 {
            print("PixelFormat \(pixelFormat) or PlaneCount \(planeCount) is not supported")
            return
        }
        
        let bytes = Int64(CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer!, 0) * CVPixelBufferGetHeightOfPlane(pixelBuffer!, 0) + CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer!, 1) * CVPixelBufferGetHeightOfPlane(pixelBuffer!, 1))
        let width = Int64(CVPixelBufferGetWidth(pixelBuffer!))
        let height = Int64(CVPixelBufferGetHeight(pixelBuffer!))
        
        if (self.bytes == 0 || self.bytes == 0 || self.height == 0) {
            self.bytes = bytes
            self.width = width
            self.height = height
        } else if (self.bytes != bytes || self.width != width || self.height != height) {
            print("Bytes \(bytes) vs. \(self.bytes), Width \(width) vs. \(self.width), Height \(height) vs. \(self.height)")
            return
        }
        if (self.buffer != nil) {
            CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            
            let baseAddress0 = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer!, 0)
            let bytesPerRow0 = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer!, 0)
            let heightOfPlane0 = CVPixelBufferGetHeightOfPlane(pixelBuffer!, 0)
            self.buffer!.copyMemory(from: baseAddress0!, byteCount: bytesPerRow0 * heightOfPlane0)
            
            let baseAddress1 = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer!, 1)
            let bytesPerRow1 = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer!, 1)
            let heightOfPlane1 = CVPixelBufferGetHeightOfPlane(pixelBuffer!, 1)
            
            self.buffer!.advanced(by: bytesPerRow0 * Int(height)).copyMemory(from: baseAddress1!, byteCount: bytesPerRow1 * heightOfPlane1)
            
            CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            
            self.copied = Int64(bytesPerRow0 * heightOfPlane0 + bytesPerRow1 * heightOfPlane1)
        }
    }
}

typealias VideoContext = (session: AVCaptureSession, semaphore_in: DispatchSemaphore, semaphore_out: DispatchSemaphore, delegate: VideoDataOutputSampleBufferDelegate)

@_silgen_name("VideoCaptureInitFunction")
func VideoCaptureInitFunction(devname: UnsafePointer<CChar>, bytes: UnsafeMutablePointer<Int64>, width: UnsafeMutablePointer<Int64>, height: UnsafeMutablePointer<Int64>) -> UnsafeMutablePointer<VideoContext>?  {
    
    let deviceName = String(cString: devname)
    
    let session = AVCaptureSession()
    let semaphore_in = DispatchSemaphore(value: 0)
    let semaphore_out = DispatchSemaphore(value: 0)
    let sampleBufferDelegate = VideoDataOutputSampleBufferDelegate(semaphore_in: semaphore_in, semaphore_out: semaphore_out)
    
    do {
        let device = AVCaptureDevice.default(for: .video)
        let deviceInput = try AVCaptureDeviceInput(device: device!)
        
        session.addInput(deviceInput)
    } catch {
        return nil
    }
    
    let queue = DispatchQueue(label: "VideoDataOutput", attributes: [])
    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [:]
    output.alwaysDiscardsLateVideoFrames = true
    output.setSampleBufferDelegate(sampleBufferDelegate, queue: queue)
    
    session.addOutput(output)
    session.commitConfiguration()
    session.startRunning()
    
    // Obtain the first frame to get the information
    semaphore_in.signal()
    semaphore_out.wait()
    
    if (sampleBufferDelegate.bytes == 0 || sampleBufferDelegate.width == 0 || sampleBufferDelegate.height == 0) {
        return nil
    }
    bytes.pointee = sampleBufferDelegate.bytes
    width.pointee = sampleBufferDelegate.width
    height.pointee = sampleBufferDelegate.height
    
    let context = UnsafeMutablePointer<VideoContext>.allocate(capacity: 1)
    context.initialize(to: (session: session, semaphore_in: semaphore_in, semaphore_out: semaphore_out, delegate: sampleBufferDelegate))
    
    return context
}

@_silgen_name("VideoCaptureNextFunction")
func VideoCaptureNextFunction(context: UnsafeMutablePointer<VideoContext>, data: UnsafeMutableRawPointer, size: Int64) -> Void {
    if context != nil {
        if (size < context.pointee.delegate.bytes) {
            print("not enough buffer to copy: \(size) vs. \(context.pointee.delegate.bytes)")
            return
        }
        context.pointee.delegate.buffer = data
        context.pointee.delegate.copied = 0
        context.pointee.semaphore_in.signal()
        context.pointee.semaphore_out.wait()
        context.pointee.delegate.buffer = nil
        if context.pointee.delegate.copied != context.pointee.delegate.bytes {
            print("not enough buffer copied: \(context.pointee.delegate.copied) vs. \(context.pointee.delegate.bytes)")
        }
        context.pointee.delegate.copied = 0
        return
    }
}

@_silgen_name("VideoCaptureFiniFunction")
func VideoCaptureFiniFunction(context: UnsafeMutablePointer<VideoContext>) -> Void {
    if context != nil {
        context.pointee.session.stopRunning()
        context.deinitialize(count: 1)
        context.deallocate()
    }
}

typealias AVCContext = (session: VTDecompressionSession, formatDescription : CMVideoFormatDescription, width: Int64, height: Int64)

@_silgen_name("DecodeAVCFunctionInit")
func DecodeAVCFunctionInit(data_pps: UnsafePointer<UInt8>, size_pps: Int64, data_sps: UnsafePointer<UInt8>, size_sps: Int64, width: UnsafeMutablePointer<Int64>, height: UnsafeMutablePointer<Int64>, bytes: UnsafeMutablePointer<Int64>) -> UnsafeMutablePointer<AVCContext>?  {
    
    let parameterSetSizes : [Int] = [ Int(size_pps), Int(size_sps) ]
    let parameterSetPointers : [UnsafePointer<UInt8>] = [ data_pps, data_sps ]
    
    var formatDescription : CMVideoFormatDescription?
    let v = CMVideoFormatDescriptionCreateFromH264ParameterSets(allocator: nil, parameterSetCount: 2, parameterSetPointers: parameterSetPointers, parameterSetSizes: parameterSetSizes, nalUnitHeaderLength: 4, formatDescriptionOut: &formatDescription )
    
    width.pointee = Int64(formatDescription!.dimensions.width)
    height.pointee = Int64(formatDescription!.dimensions.height)
    bytes.pointee = Int64(formatDescription!.dimensions.width * formatDescription!.dimensions.height * 3 / 2)
    
    var session: VTDecompressionSession? = nil
    
    let status = VTDecompressionSessionCreate(allocator: nil, formatDescription: formatDescription!, decoderSpecification: nil, imageBufferAttributes: nil, outputCallback: nil, decompressionSessionOut: &session)
    
    let context = UnsafeMutablePointer<AVCContext>.allocate(capacity: 1)
    context.initialize(to: (session: session!, formatDescription: formatDescription!, width: width.pointee, height: height.pointee))
    
    return context
    
    
}

@_silgen_name("DecodeAVCFunctionNext")
func DecodeAVCFunctionNext(context: UnsafeMutablePointer<AVCContext>, data_in: UnsafeRawPointer, size_in: Int64, data_out: UnsafeMutableRawPointer, size_out: Int64) -> Int64 {
    
    let blockBuffer = try! CMBlockBuffer(length: Int(size_in))
    
    CMBlockBufferReplaceDataBytes(with: data_in, blockBuffer: blockBuffer, offsetIntoDestination: 0, dataLength: Int(size_in))
    
    let sampleBuffer = try! CMSampleBuffer(dataBuffer: blockBuffer, formatDescription: context.pointee.formatDescription, numSamples: 1, sampleTimings: [], sampleSizes: [1])
    
    func outputHandler(status: OSStatus, infoFlags: VTDecodeInfoFlags, buffer: CVImageBuffer?, t1: CMTime, t2: CMTime) -> Void {
        
        let bytesPerRow0 = Int(context.pointee.width)
        let bytesPaddedPerRow0 = CVPixelBufferGetBytesPerRowOfPlane(buffer!, 0)
        let heightOfPlane0 = CVPixelBufferGetHeightOfPlane(buffer!, 0)
        
        let bytesPerRow1 = Int(context.pointee.width)
        let bytesPaddedPerRow1 = CVPixelBufferGetBytesPerRowOfPlane(buffer!, 1)
        let heightOfPlane1 = CVPixelBufferGetHeightOfPlane(buffer!, 1)
        
        if size_out < bytesPerRow0 * heightOfPlane0 + bytesPerRow1 * heightOfPlane1 {
            return
        }
        
        CVPixelBufferLockBaseAddress(buffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        let baseAddress0 = CVPixelBufferGetBaseAddressOfPlane(buffer!, 0)
        
        var bytesAdvanced0 = 0
        var bytesPaddedAdvanced0 = 0
        for _ in 1...heightOfPlane0 {
            data_out.advanced(by: bytesAdvanced0).copyMemory(from: baseAddress0!.advanced(by: bytesPaddedAdvanced0), byteCount: bytesPerRow0)
            
            bytesAdvanced0 += bytesPerRow0
            bytesPaddedAdvanced0 += bytesPaddedPerRow0
        }
        
        let baseAddress1 = CVPixelBufferGetBaseAddressOfPlane(buffer!, 1)
        
        var bytesAdvanced1 = bytesAdvanced0
        var bytesPaddedAdvanced1 = 0
        for _ in 1...heightOfPlane1 {
            data_out.advanced(by: bytesAdvanced1).copyMemory(from: baseAddress1!.advanced(by: bytesPaddedAdvanced1), byteCount: bytesPerRow1)
            
            bytesAdvanced1 += bytesPerRow1
            bytesPaddedAdvanced1 += bytesPaddedPerRow1
        }
        
        CVPixelBufferUnlockBaseAddress(buffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return
    }
    VTDecompressionSessionDecodeFrame(context.pointee.session, sampleBuffer: sampleBuffer, flags: [], infoFlagsOut: nil, outputHandler: outputHandler)
    
    return 0
}

@_silgen_name("DecodeAVCFunctionFini")
func DecodeAVCFunctionFini(context: UnsafeMutablePointer<AVCContext>) -> Void {
    context.deinitialize(count: 1)
    context.deallocate()
}
