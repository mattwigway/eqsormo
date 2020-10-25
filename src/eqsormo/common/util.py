#    Copyright 2020 Matthew Wigginton Conway

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

byte_scalars = {"TB": 1e12, "GB": 1e9, "MB": 1e6, "KB": 1e3}


def human_bytes(nbytes):
    "Return bytes in human-readable format, e.g. 1.42GB"
    for suffix, scalar in byte_scalars.items():
        if nbytes > scalar:
            return f"{nbytes / scalar:.2f} {suffix}"

    return f"{nbytes} bytes"


def human_time(seconds):
    "Time in human readable format, e.g. 3630 -> 1h0m30s"
    ret = []
    if seconds >= 3600:
        ret.append(f"{int(seconds // 3600)}h")
        seconds = seconds % 3600
    if seconds >= 60:
        ret.append(f"{int(seconds // 60)}m")
        seconds = seconds % 60
    ret.append(f"{seconds:.3f}s")

    return " ".join(ret)


def human_shape(shape):
    "Shape of a numpy array in human-readable format, e.g. 2x2"
    return "x".join(map(str, shape))
