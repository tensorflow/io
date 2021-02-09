/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#if defined(__linux__)

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int xioctl(int fh, int request, void* arg) {
  int r;

  do {
    r = ioctl(fh, request, arg);
  } while (-1 == r && EINTR == errno);

  return r;
}
namespace tensorflow {
namespace data {

class VideoCaptureContext {
 public:
  VideoCaptureContext()
      : context_(nullptr,
                 [](void* p) {
                   if (p != nullptr) {
                     free(p);
                   }
                 }),
        fd_scope_(nullptr, [](int* p) {
          if (p != nullptr) {
            close(*p);
          }
        }) {}
  ~VideoCaptureContext() {}

  Status Init(const string& device, int64_t* bytes, int64_t* width,
              int64_t* height) {
    device_ = device;

    const char* devname = device.c_str();
    struct stat st;
    if (-1 == stat(devname, &st)) {
      return errors::InvalidArgument("cannot identify '", devname, "': ", errno,
                                     ", ", strerror(errno));
    }

    if (!S_ISCHR(st.st_mode)) {
      return errors::InvalidArgument(devname, " is no device");
    }

    fd_ = open(devname, O_RDWR /* required */ | O_NONBLOCK, 0);
    if (-1 == fd_) {
      return errors::InvalidArgument("cannot open '", devname, "': ", errno,
                                     ", ", strerror(errno));
    }
    fd_scope_.reset(&fd_);

    struct v4l2_capability cap;
    if (-1 == xioctl(fd_, VIDIOC_QUERYCAP, &cap)) {
      if (EINVAL == errno) {
        return errors::InvalidArgument(devname, " is no V4L2 device");
      } else {
        return errors::InvalidArgument("cannot VIDIOC_QUERYCAP '", devname,
                                       "': ", errno, ", ", strerror(errno));
      }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      return errors::InvalidArgument(devname, " is no video capture device");
    }

    if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
      return errors::InvalidArgument(devname, " does not support read i/o");
    }

    struct v4l2_format fmt;
    memset(&(fmt), 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd_, VIDIOC_G_FMT, &fmt)) {
      return errors::InvalidArgument("cannot VIDIOC_G_FMT '", devname,
                                     "': ", errno, ", ", strerror(errno));
    }

    /* Buggy driver paranoia. */
    {
      unsigned int min;
      min = fmt.fmt.pix.width * 2;
      if (fmt.fmt.pix.bytesperline < min) {
        fmt.fmt.pix.bytesperline = min;
      }
      min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
      if (fmt.fmt.pix.sizeimage < min) {
        fmt.fmt.pix.sizeimage = min;
      }
    }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
      return errors::InvalidArgument(
          "only V4L2_PIX_FMT_YUYV is supported, received ",
          fmt.fmt.pix.pixelformat);
    }

    *bytes = fmt.fmt.pix.sizeimage;
    *width = fmt.fmt.pix.width;
    *height = fmt.fmt.pix.height;

    return Status::OK();
  }
  Status Read(void* data, size_t size) {
    do {
      fd_set fds;
      struct timeval tv;
      int r;

      FD_ZERO(&fds);
      FD_SET(fd_, &fds);

      /* Timeout. */
      tv.tv_sec = 2;
      tv.tv_usec = 0;
      r = select(fd_ + 1, &fds, NULL, NULL, &tv);

      if (-1 == r) {
        if (EINTR == errno) {
          continue;
        }
        return errors::InvalidArgument("cannot select: ", errno, ", ",
                                       strerror(errno));
      }
      if (0 == r) {
        return errors::InvalidArgument("select timeout");
      }

      if (-1 == read(fd_, data, size)) {
        if (EAGAIN == errno) {
          /* EAGAIN - continue select loop. */
          continue;
        }
        if (EIO == errno) {
          /* Could ignore EIO, see spec. */
          /* fall through */
        }
        return errors::InvalidArgument("cannot read: ", errno, ", ",
                                       strerror(errno));
      }
      // Data Obtained, break
      break;
    } while (true);
    return Status::OK();
  }

 protected:
  mutable mutex mu_;

  std::unique_ptr<void, void (*)(void*)> context_;
  std::unique_ptr<int, void (*)(int*)> fd_scope_;
  string device_;
  int fd_;
};

}  // namespace data
}  // namespace tensorflow
#endif
