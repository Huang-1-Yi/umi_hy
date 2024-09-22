
# imagecodecs/numcodecs.py

# Copyright (c) 2021-2022, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Additional numcodecs implemented using imagecodecs."""
"""使用图像编解码器实现的附加数字编解码器。"""

__version__ = '2022.9.26'

__all__ = ('register_codecs',)

import numpy                        # 导入numpy库
from numcodecs.abc import Codec     # 从numcodecs.abc导入Codec类
from numcodecs.registry import register_codec, get_codec  # 从numcodecs.registry导入register_codec和get_codec函数
import imagecodecs                  # 导入imagecodecs库


def protective_squeeze(x: numpy.ndarray):
    """
    Squeeze dim only if it's not the last dim.”仅当不是最后一次变暗时才挤压变暗
    Image dim expected to be *, H, W, C图像暗淡预计为 *、H、W、C
    """
    img_shape = x.shape[-3:]        # 获取图像的最后三个维度
    if len(x.shape) > 3:            # 如果维度超过3
        n_imgs = numpy.prod(x.shape[:-3])   # 计算图像数量
        if n_imgs > 1:              # 如果图像数量大于1,调整形状
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)     # 重新调整形状并返回

def get_default_image_compressor(**kwargs):
    if imagecodecs.JPEGXL:          # 如果支持JPEGXL
        this_kwargs = {
            'effort': 3,
            'distance': 0.3,
            # libjxl中的一个bug，当解码速度大于1时，非无损的情况下无效码流
            'decodingspeed': 1
        }
        this_kwargs.update(kwargs)  # 更新参数
        return JpegXl(**this_kwargs)# 返回JpegXl实例
    else:
        this_kwargs = {
            'level': 50
        }
        this_kwargs.update(kwargs)  # 更新参数
        return Jpeg2k(**this_kwargs)# 返回Jpeg2k实例

class Aec(Codec):
    """AEC codec for numcodecs.AEC 编解码器"""

    codec_id = 'imagecodecs_aec'

    def __init__(
        self, bitspersample=None, flags=None, blocksize=None, rsi=None
    ):
        self.bitspersample = bitspersample  # 位样本数
        self.flags = flags  # 编码标志
        self.blocksize = blocksize  # 块大小
        self.rsi = rsi  # RSI参数

    def encode(self, buf):
        return imagecodecs.aec_encode(
            buf,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.aec_decode(
            buf,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
            out=_flat(out),
        )  # 解码并返回结果

class Apng(Codec):
    """APNG codec for numcodecs."""

    codec_id = 'imagecodecs_apng'

    def __init__(self, level=None, photometric=None, delay=None):
        self.level = level  # 编码等级
        self.photometric = photometric  # 光度参数
        self.delay = delay  # 延迟参数

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.apng_encode(
            buf,
            level=self.level,
            photometric=self.photometric,
            delay=self.delay,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.apng_decode(buf, out=out)  # 解码并返回结果



class Avif(Codec):
    """AVIF codec for numcodecs."""

    codec_id = 'imagecodecs_avif'

    def __init__(
        self,
        level=None,
        speed=None,
        tilelog2=None,
        bitspersample=None,
        pixelformat=None,
        numthreads=None,
        index=None,
    ):
        self.level = level  # 编码等级
        self.speed = speed  # 编码速度
        self.tilelog2 = tilelog2  # 瓦片参数
        self.bitspersample = bitspersample  # 位样本数
        self.pixelformat = pixelformat  # 像素格式
        self.numthreads = numthreads  # 线程数
        self.index = index  # 索引

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.avif_encode(
            buf,
            level=self.level,
            speed=self.speed,
            tilelog2=self.tilelog2,
            bitspersample=self.bitspersample,
            pixelformat=self.pixelformat,
            numthreads=self.numthreads,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.avif_decode(
            buf, index=self.index, numthreads=self.numthreads, out=out
        )  # 解码并返回结果


class Bitorder(Codec):
    """Bitorder codec for numcodecs."""

    codec_id = 'imagecodecs_bitorder'

    def encode(self, buf):
        return imagecodecs.bitorder_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.bitorder_decode(buf, out=_flat(out))  # 解码并返回结果


class Bitshuffle(Codec):
    """Bitshuffle codec for numcodecs."""

    codec_id = 'imagecodecs_bitshuffle'

    def __init__(self, itemsize=1, blocksize=0):
        self.itemsize = itemsize    # 项目大小
        self.blocksize = blocksize  # 块大小

    def encode(self, buf):
        return imagecodecs.bitshuffle_encode(
            buf, itemsize=self.itemsize, blocksize=self.blocksize
        ).tobytes()  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.bitshuffle_decode(
            buf,
            itemsize=self.itemsize,
            blocksize=self.blocksize,
            out=_flat(out),
        )  # 解码并返回结果


class Blosc(Codec):
    """Blosc codec for numcodecs."""

    codec_id = 'imagecodecs_blosc'

    def __init__(
        self,
        level=None,
        compressor=None,
        typesize=None,
        blocksize=None,
        shuffle=None,
        numthreads=None,
    ):
        self.level = level  # 编码等级
        self.compressor = compressor  # 压缩器类型
        self.typesize = typesize  # 类型大小
        self.blocksize = blocksize  # 块大小
        self.shuffle = shuffle  # shuffle参数
        self.numthreads = numthreads  # 线程数

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.blosc_encode(
            buf,
            level=self.level,
            compressor=self.compressor,
            typesize=self.typesize,
            blocksize=self.blocksize,
            shuffle=self.shuffle,
            numthreads=self.numthreads,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.blosc_decode(
            buf, numthreads=self.numthreads, out=_flat(out)
        )  # 解码并返回结果


class Blosc2(Codec):
    """Blosc2 codec for numcodecs."""

    codec_id = 'imagecodecs_blosc2'

    def __init__(
        self,
        level=None,
        compressor=None,
        typesize=None,
        blocksize=None,
        shuffle=None,
        numthreads=None,
    ):
        self.level = level  # 编码等级
        self.compressor = compressor  # 压缩器类型
        self.typesize = typesize  # 类型大小
        self.blocksize = blocksize  # 块大小
        self.shuffle = shuffle  # shuffle参数
        self.numthreads = numthreads  # 线程数

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.blosc2_encode(
            buf,
            level=self.level,
            compressor=self.compressor,
            typesize=self.typesize,
            blocksize=self.blocksize,
            shuffle=self.shuffle,
            numthreads=self.numthreads,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.blosc2_decode(
            buf, numthreads=self.numthreads, out=_flat(out)
        )  # 解码并返回结果


class Brotli(Codec):
    """Brotli codec for numcodecs."""

    codec_id = 'imagecodecs_brotli'

    def __init__(self, level=None, mode=None, lgwin=None):
        self.level = level  # 编码等级
        self.mode = mode  # 编码模式
        self.lgwin = lgwin  # 窗口大小

    def encode(self, buf):
        return imagecodecs.brotli_encode(
            buf, level=self.level, mode=self.mode, lgwin=self.lgwin
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.brotli_decode(buf, out=_flat(out))  # 解码并返回结果


class ByteShuffle(Codec):
    """ByteShuffle codec for numcodecs."""

    codec_id = 'imagecodecs_byteshuffle'

    def __init__(
        self, shape, dtype, axis=-1, dist=1, delta=False, reorder=False
    ):
        self.shape = tuple(shape)  # 形状
        self.dtype = numpy.dtype(dtype).str  # 数据类型
        self.axis = axis  # 轴
        self.dist = dist  # 距离
        self.delta = bool(delta)  # 是否使用delta
        self.reorder = bool(reorder)  # 是否重新排序

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        assert buf.shape == self.shape  # 确保形状匹配
        assert buf.dtype == self.dtype  # 确保数据类型匹配
        return imagecodecs.byteshuffle_encode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
        ).tobytes()  # 编码并返回结果

    def decode(self, buf, out=None):
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)  # 重新调整形状
        return imagecodecs.byteshuffle_decode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
            out=out,
        )  # 解码并返回结果


class Bz2(Codec):
    """Bz2编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_bz2'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        return imagecodecs.bz2_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.bz2_decode(buf, out=_flat(out))  # 解码并返回结果

class Cms(Codec):
    """CMS编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_cms'

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, buf, out=None):
        # 返回imagecodecs.cms_transform(buf)
        raise NotImplementedError  # 尚未实现

    def decode(self, buf, out=None):
        # 返回imagecodecs.cms_transform(buf)
        raise NotImplementedError  # 尚未实现

class Deflate(Codec):
    """Deflate编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_deflate'

    def __init__(self, level=None, raw=False):
        self.level = level  # 编码等级
        self.raw = bool(raw)  # 是否为原始数据

    def encode(self, buf):
        return imagecodecs.deflate_encode(buf, level=self.level, raw=self.raw)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.deflate_decode(buf, out=_flat(out), raw=self.raw)  # 解码并返回结果

class Delta(Codec):
    """Delta编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_delta'

    def __init__(self, shape=None, dtype=None, axis=-1, dist=1):
        self.shape = None if shape is None else tuple(shape)  # 形状
        self.dtype = None if dtype is None else numpy.dtype(dtype).str  # 数据类型
        self.axis = axis  # 轴
        self.dist = dist  # 距离

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
            assert buf.shape == self.shape  # 确保形状匹配
            assert buf.dtype == self.dtype  # 确保数据类型匹配
        return imagecodecs.delta_encode(
            buf, axis=self.axis, dist=self.dist
        ).tobytes()  # 编码并返回结果

    def decode(self, buf, out=None):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)  # 重新调整形状
        return imagecodecs.delta_decode(
            buf, axis=self.axis, dist=self.dist, out=out
        )  # 解码并返回结果

class Float24(Codec):
    """Float24编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_float24'

    def __init__(self, byteorder=None, rounding=None):
        self.byteorder = byteorder  # 字节顺序
        self.rounding = rounding  # 四舍五入

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.float24_encode(
            buf, byteorder=self.byteorder, rounding=self.rounding
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.float24_decode(
            buf, byteorder=self.byteorder, out=out
        )  # 解码并返回结果

class FloatPred(Codec):
    """浮点预测编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_floatpred'

    def __init__(self, shape, dtype, axis=-1, dist=1):
        self.shape = tuple(shape)  # 形状
        self.dtype = numpy.dtype(dtype).str  # 数据类型
        self.axis = axis  # 轴
        self.dist = dist  # 距离

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        assert buf.shape == self.shape  # 确保形状匹配
        assert buf.dtype == self.dtype  # 确保数据类型匹配
        return imagecodecs.floatpred_encode(
            buf, axis=self.axis, dist=self.dist
        ).tobytes()  # 编码并返回结果

    def decode(self, buf, out=None):
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)  # 重新调整形状
        return imagecodecs.floatpred_decode(
            buf, axis=self.axis, dist=self.dist, out=out
        )  # 解码并返回结果

class Gif(Codec):
    """GIF编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_gif'

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.gif_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.gif_decode(buf, asrgb=False, out=out)  # 解码并返回结果

class Heif(Codec):
    """HEIF编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_heif'

    def __init__(
        self,
        level=None,
        bitspersample=None,
        photometric=None,
        compression=None,
        numthreads=None,
        index=None,
    ):
        self.level = level  # 编码等级
        self.bitspersample = bitspersample  # 位样本数
        self.photometric = photometric  # 光度参数
        self.compression = compression  # 压缩参数
        self.numthreads = numthreads  # 线程数
        self.index = index  # 索引

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.heif_encode(
            buf,
            level=self.level,
            bitspersample=self.bitspersample,
            photometric=self.photometric,
            compression=self.compression,
            numthreads=self.numthreads,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.heif_decode(
            buf,
            index=self.index,
            photometric=self.photometric,
            numthreads=self.numthreads,
            out=out,
        )  # 解码并返回结果


class Jetraw(Codec):
    """Jetraw编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_jetraw'

    def __init__(
        self,
        shape,
        identifier,
        parameters=None,
        verbosity=None,
        errorbound=None,
    ):
        self.shape = shape  # 形状
        self.identifier = identifier  # 标识符
        self.errorbound = errorbound  # 错误界限
        imagecodecs.jetraw_init(parameters, verbosity)  # 初始化Jetraw

    def encode(self, buf):
        return imagecodecs.jetraw_encode(
            buf, identifier=self.identifier, errorbound=self.errorbound
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        if out is None:
            out = numpy.empty(self.shape, numpy.uint16)  # 创建空数组
        return imagecodecs.jetraw_decode(buf, out=out)  # 解码并返回结果

class Jpeg(Codec):
    """JPEG编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_jpeg'

    def __init__(
        self,
        bitspersample=None,
        tables=None,
        header=None,
        colorspace_data=None,
        colorspace_jpeg=None,
        level=None,
        subsampling=None,
        optimize=None,
        smoothing=None,
    ):
        self.tables = tables  # 表
        self.header = header  # 头部
        self.bitspersample = bitspersample  # 位样本数
        self.colorspace_data = colorspace_data  # 数据颜色空间
        self.colorspace_jpeg = colorspace_jpeg  # JPEG颜色空间
        self.level = level  # 编码等级
        self.subsampling = subsampling  # 采样
        self.optimize = optimize  # 优化参数
        self.smoothing = smoothing  # 平滑参数

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.jpeg_encode(
            buf,
            level=self.level,
            colorspace=self.colorspace_data,
            outcolorspace=self.colorspace_jpeg,
            subsampling=self.subsampling,
            optimize=self.optimize,
            smoothing=self.smoothing,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        out_shape = None
        if out is not None:
            out_shape = out.shape  # 获取输出形状
            out = protective_squeeze(out)  # 压缩输出
        img = imagecodecs.jpeg_decode(
            buf,
            bitspersample=self.bitspersample,
            tables=self.tables,
            header=self.header,
            colorspace=self.colorspace_jpeg,
            outcolorspace=self.colorspace_data,
            out=out,
        )  # 解码图像
        if out_shape is not None:
            img = img.reshape(out_shape)  # 重新调整形状
        return img  # 返回图像

    def get_config(self):
        """返回包含配置参数的字典。"""
        config = dict(id=self.codec_id)  # 创建配置字典
        for key in self.__dict__:
            if not key.startswith('_'):
                value = getattr(self, key)
                if value is not None and key in ('header', 'tables'):
                    import base64

                    value = base64.b64encode(value).decode()  # 进行Base64编码
                config[key] = value
        return config  # 返回配置字典

    @classmethod
    def from_config(cls, config):
        """从配置对象实例化编解码器。"""
        for key in ('header', 'tables'):
            value = config.get(key, None)
            if value is not None and isinstance(value, str):
                import base64

                config[key] = base64.b64decode(value.encode())  # 进行Base64解码
        return cls(**config)  # 返回实例

class Jpeg2k(Codec):
    """JPEG 2000编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_jpeg2k'

    def __init__(
        self,
        level=None,
        codecformat=None,
        colorspace=None,
        tile=None,
        reversible=None,
        bitspersample=None,
        resolutions=None,
        numthreads=None,
        verbose=0,
    ):
        self.level = level  # 编码等级
        self.codecformat = codecformat  # 编解码格式
        self.colorspace = colorspace  # 颜色空间
        self.tile = None if tile is None else tuple(tile)  # 瓦片
        self.reversible = reversible  # 是否可逆
        self.bitspersample = bitspersample  # 位样本数
        self.resolutions = resolutions  # 分辨率
        self.numthreads = numthreads  # 线程数
        self.verbose = verbose  # 详细程度

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.jpeg2k_encode(
            buf,
            level=self.level,
            codecformat=self.codecformat,
            colorspace=self.colorspace,
            tile=self.tile,
            reversible=self.reversible,
            bitspersample=self.bitspersample,
            resolutions=self.resolutions,
            numthreads=self.numthreads,
            verbose=self.verbose,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.jpeg2k_decode(
            buf, verbose=self.verbose, numthreads=self.numthreads, out=out
        )  # 解码并返回结果

class JpegLs(Codec):
    """JPEG LS编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_jpegls'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.jpegls_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.jpegls_decode(buf, out=out)  # 解码并返回结果


class JpegXl(Codec):
    """JPEG XL codec for numcodecs."""

    codec_id = 'imagecodecs_jpegxl'

    def __init__(
        self,
        # 编码参数
        level=None,
        effort=None,
        distance=None,
        lossless=None,
        decodingspeed=None,
        photometric=None,
        planar=None,
        usecontainer=None,
        # 解码参数
        index=None,
        keeporientation=None,
        # 通用参数
        numthreads=None,
    ):
        """
        从numpy数组返回JPEG XL图像。
        浮点数必须在0到1的标称范围内。

        当前支持L, LA, RGB, RGBA图像。
        仅在平面模式下支持灰度图像的额外通道。

        参数
        ----------
        level : 默认为None，即不覆盖lossess和decodingspeed选项。
            当< 0时：使用无损压缩
            当在[0,1,2,3,4]之间时：设置提供选项的解码速度级别。最小值为0（解码速度最慢，质量/密度最佳），最大值为4（解码速度最快，以某些质量/密度为代价）。
        effort : 默认为3。
            设置编码器努力/速度级别，不影响解码速度。有效值为，从快到慢：1:lightning 2:thunder 3:falcon 4:cheetah 5:hare 6:wombat 7:squirrel 8:kitten 9:tortoise。
            速度：lightning, thunder, falcon, cheetah, hare, wombat, squirrel, kitten, tortoise 控制编码器的努力按升序。
            这也会影响内存使用：使用较低的努力通常会减少编码期间的内存消耗。
            lightning和thunder是用于无损模式（模块化）的快速模式。
            falcon禁用以下所有工具。
            cheetah启用系数重新排序、上下文聚类和用于选择DCT大小和量化步骤的启发式方法。
            hare启用Gaborish滤波、来自luma的色度和量化步骤的初步估计。
            wombat启用误差扩散量化和完整的DCT大小选择启发式方法。
            squirrel（默认）启用点、补丁和样条检测以及完整的上下文聚类。
            kitten优化了自适应量化以适应心理视觉度量。
            tortoise启用更彻底的自适应量化搜索。
        distance : 默认为1.0
            设置有损压缩的距离级别：目标最大butteraugli距离，值越低质量越高。范围：0 .. 15。0.0 = 数学无损（但是，使用JxlEncoderSetFrameLossless而不是单独设置距离为0是使用真正无损的要求）。1.0 = 视觉无损。推荐范围：0.5 .. 3.0。
        lossess : 默认为False。
            使用无损编码。
        decodingspeed : 默认为0。
            复制到level。 [0,4]
        photometric : 返回JxlColorSpace值。
            默认逻辑非常复杂，但大多数时候有效。
            接受的值：
                int: [-1,3]
                str: ['RGB', 'WHITEISZERO', 'MINISWHITE', 'BLACKISZERO', 'MINISBLACK', 'GRAY', 'XYB', 'KNOWN']
        planar : 启用多通道模式。
            默认为false。
        usecontainer : 强制编码器使用基于盒的容器格式（BMFF），即使不必要。
            使用JxlEncoderUseBoxes、JxlEncoderStoreJPEGMetadata或使用级别10的JxlEncoderSetCodestreamLevel时，编码器将自动使用容器格式，不需要使用JxlEncoderUseContainer。
            默认情况下，此设置被禁用。
        index : 选择性解码动画帧。
            默认为0，解码所有帧。
            当设置为> 0时，仅解码该帧索引。
        keeporientation : 启用或禁用保留按位流像素数据方向。
            某些图像使用Orientation标记进行编码，指示解码器必须对编码图像数据进行旋转和/或镜像。

            如果skip_reorientation为JXL_FALSE（默认值）：解码器将应用方向设置的转换，从而根据其指定的意图渲染图像。生成JxlBasicInfo时，解码器将始终将方向字段设置为JXL_ORIENT_IDENTITY（与返回的像素数据匹配），并对xsize和ysize进行对齐，使它们对应于返回的像素数据的宽度和高度。

            如果skip_reorientation为JXL_TRUE：解码器将跳过应用方向设置的转换，返回按位流像素数据方向的图像。这可能更快地解码，因为解码器不必应用转换，但如果用户没有正确考虑方向标记，可能会导致图像显示错误。

            默认情况下，此选项被禁用，并且返回的像素数据将根据图像的Orientation设置重新定向。
        threads : 默认为1。
            如果<= 0，使用所有内核。
            如果> 32，则限制为32。
        """

        self.level = level  # 编码等级
        self.effort = effort  # 努力级别
        self.distance = distance  # 距离级别
        self.lossless = bool(lossless)  # 是否无损
        self.decodingspeed = decodingspeed  # 解码速度
        self.photometric = photometric  # 光度参数
        self.planar = planar  # 平面模式
        self.usecontainer = usecontainer  # 使用容器格式
        self.index = index  # 索引
        self.keeporientation = keeporientation  # 保持方向
        self.numthreads = numthreads  # 线程数

    def encode(self, buf):
        # TODO: only squeeze all but last dim
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.jpegxl_encode(
            buf,
            level=self.level,
            effort=self.effort,
            distance=self.distance,
            lossless=self.lossless,
            decodingspeed=self.decodingspeed,
            photometric=self.photometric,
            planar=self.planar,
            usecontainer=self.usecontainer,
            numthreads=self.numthreads,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.jpegxl_decode(
            buf,
            index=self.index,
            keeporientation=self.keeporientation,
            numthreads=self.numthreads,
            out=out,
        )  # 解码并返回结果


class JpegXr(Codec):
    """JPEG XR编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_jpegxr'

    def __init__(
        self,
        level=None,
        photometric=None,
        hasalpha=None,
        resolution=None,
        fp2int=None,
    ):
        self.level = level  # 编码等级
        self.photometric = photometric  # 光度参数
        self.hasalpha = hasalpha  # 是否有Alpha通道
        self.resolution = resolution  # 分辨率
        self.fp2int = fp2int  # 浮点到整数

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.jpegxr_encode(
            buf,
            level=self.level,
            photometric=self.photometric,
            hasalpha=self.hasalpha,
            resolution=self.resolution,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.jpegxr_decode(buf, fp2int=self.fp2int, out=out)  # 解码并返回结果

class Lerc(Codec):
    """LERC编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_lerc'

    def __init__(self, level=None, version=None, planar=None):
        self.level = level  # 编码等级
        self.version = version  # 版本
        self.planar = bool(planar)  # 是否平面模式
        # TODO: 支持mask
        # self.mask = None

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.lerc_encode(
            buf,
            level=self.level,
            version=self.version,
            planar=self.planar,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.lerc_decode(buf, out=out)  # 解码并返回结果

class Ljpeg(Codec):
    """LJPEG编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_ljpeg'

    def __init__(self, bitspersample=None):
        self.bitspersample = bitspersample  # 位样本数

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.ljpeg_encode(buf, bitspersample=self.bitspersample)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.ljpeg_decode(buf, out=out)  # 解码并返回结果

class Lz4(Codec):
    """LZ4编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_lz4'

    def __init__(self, level=None, hc=False, header=True):
        self.level = level  # 编码等级
        self.hc = hc  # 高压缩模式
        self.header = bool(header)  # 是否包含头部

    def encode(self, buf):
        return imagecodecs.lz4_encode(
            buf, level=self.level, hc=self.hc, header=self.header
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.lz4_decode(buf, header=self.header, out=_flat(out))  # 解码并返回结果

class Lz4f(Codec):
    """LZ4F编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_lz4f'

    def __init__(
        self,
        level=None,
        blocksizeid=False,
        contentchecksum=None,
        blockchecksum=None,
    ):
        self.level = level  # 编码等级
        self.blocksizeid = blocksizeid  # 块大小ID
        self.contentchecksum = contentchecksum  # 内容校验和
        self.blockchecksum = blockchecksum  # 块校验和

    def encode(self, buf):
        return imagecodecs.lz4f_encode(
            buf,
            level=self.level,
            blocksizeid=self.blocksizeid,
            contentchecksum=self.contentchecksum,
            blockchecksum=self.blockchecksum,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.lz4f_decode(buf, out=_flat(out))  # 解码并返回结果

class Lzf(Codec):
    """LZF编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_lzf'

    def __init__(self, header=True):
        self.header = bool(header)  # 是否包含头部

    def encode(self, buf):
        return imagecodecs.lzf_encode(buf, header=self.header)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.lzf_decode(buf, header=self.header, out=_flat(out))  # 解码并返回结果

class Lzma(Codec):
    """LZMA编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_lzma'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        return imagecodecs.lzma_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.lzma_decode(buf, out=_flat(out))  # 解码并返回结果

class Lzw(Codec):
    """LZW编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_lzw'

    def encode(self, buf):
        return imagecodecs.lzw_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.lzw_decode(buf, out=_flat(out))  # 解码并返回结果



class PackBits(Codec):
    """PackBits编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_packbits'

    def __init__(self, axis=None):
        self.axis = axis  # 轴

    def encode(self, buf):
        if not isinstance(buf, (bytes, bytearray)):
            buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.packbits_encode(buf, axis=self.axis)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.packbits_decode(buf, out=_flat(out))  # 解码并返回结果

class Pglz(Codec):
    """PGLZ编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_pglz'

    def __init__(self, header=True, strategy=None):
        self.header = bool(header)  # 是否包含头部
        self.strategy = strategy  # 策略

    def encode(self, buf):
        return imagecodecs.pglz_encode(
            buf, strategy=self.strategy, header=self.header
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.pglz_decode(buf, header=self.header, out=_flat(out))  # 解码并返回结果

class Png(Codec):
    """PNG编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_png'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.png_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.png_decode(buf, out=out)  # 解码并返回结果

class Qoi(Codec):
    """QOI编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_qoi'

    def __init__(self):
        pass

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.qoi_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.qoi_decode(buf, out=out)  # 解码并返回结果

class Rgbe(Codec):
    """RGBE编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_rgbe'

    def __init__(self, header=False, shape=None, rle=None):
        if not header and shape is None:
            raise ValueError('must specify data shape if no header')  # 如果没有头部且没有指定形状，则引发错误
        if shape and shape[-1] != 3:
            raise ValueError('invalid shape')  # 如果形状无效，则引发错误
        self.shape = shape  # 形状
        self.header = bool(header)  # 是否包含头部
        self.rle = None if rle is None else bool(rle)  # 是否使用RLE

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.rgbe_encode(buf, header=self.header, rle=self.rle)  # 编码并返回结果

    def decode(self, buf, out=None):
        if out is None and not self.header:
            out = numpy.empty(self.shape, numpy.float32)  # 创建空数组
        return imagecodecs.rgbe_decode(
            buf, header=self.header, rle=self.rle, out=out
        )  # 解码并返回结果

class Rcomp(Codec):
    """Rcomp编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_rcomp'

    def __init__(self, shape, dtype, nblock=None):
        self.shape = tuple(shape)  # 形状
        self.dtype = numpy.dtype(dtype).str  # 数据类型
        self.nblock = nblock  # 块数

    def encode(self, buf):
        return imagecodecs.rcomp_encode(buf, nblock=self.nblock)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.rcomp_decode(
            buf,
            shape=self.shape,
            dtype=self.dtype,
            nblock=self.nblock,
            out=out,
        )  # 解码并返回结果

class Snappy(Codec):
    """Snappy编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_snappy'

    def encode(self, buf):
        return imagecodecs.snappy_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.snappy_decode(buf, out=_flat(out))  # 解码并返回结果

class Spng(Codec):
    """SPNG编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_spng'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.spng_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.spng_decode(buf, out=out)  # 解码并返回结果

class Tiff(Codec):
    """TIFF编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_tiff'

    def __init__(self, index=None, asrgb=None, verbose=0):
        self.index = index  # 索引
        self.asrgb = bool(asrgb)  # 是否为RGB
        self.verbose = verbose  # 详细程度

    def encode(self, buf):
        # TODO: 尚未实现
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.tiff_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.tiff_decode(
            buf,
            index=self.index,
            asrgb=self.asrgb,
            verbose=self.verbose,
            out=out,
        )  # 解码并返回结果

class Webp(Codec):
    """WebP编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_webp'

    def __init__(self, level=None, lossless=None, method=None, hasalpha=None):
        self.level = level  # 编码等级
        self.hasalpha = bool(hasalpha)  # 是否有Alpha通道
        self.method = method  # 编码方法
        self.lossless = lossless  # 是否无损

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        return imagecodecs.webp_encode(
            buf, level=self.level, lossless=self.lossless, method=self.method
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.webp_decode(buf, hasalpha=self.hasalpha, out=out)  # 解码并返回结果

class Xor(Codec):
    """XOR编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_xor'

    def __init__(self, shape=None, dtype=None, axis=-1):
        self.shape = None if shape is None else tuple(shape)  # 形状
        self.dtype = None if dtype is None else numpy.dtype(dtype).str  # 数据类型
        self.axis = axis  # 轴

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
            assert buf.shape == self.shape  # 确保形状匹配
            assert buf.dtype == self.dtype  # 确保数据类型匹配
        return imagecodecs.xor_encode(buf, axis=self.axis).tobytes()  # 编码并返回结果

    def decode(self, buf, out=None):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)  # 重新调整形状
        return imagecodecs.xor_decode(buf, axis=self.axis, out=_flat(out))  # 解码并返回结果

class Zfp(Codec):
    """ZFP编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_zfp'

    def __init__(
        self,
        shape=None,
        dtype=None,
        strides=None,
        level=None,
        mode=None,
        execution=None,
        numthreads=None,
        chunksize=None,
        header=True,
    ):
        if header:
            self.shape = None
            self.dtype = None
            self.strides = None
        elif shape is None or dtype is None:
            raise ValueError('invalid shape or dtype')  # 如果形状或数据类型无效，则引发错误
        else:
            self.shape = tuple(shape)  # 形状
            self.dtype = numpy.dtype(dtype).str  # 数据类型
            self.strides = None if strides is None else tuple(strides)  # 步幅
        self.level = level  # 编码等级
        self.mode = mode  # 编码模式
        self.execution = execution  # 执行模式
        self.numthreads = numthreads  # 线程数
        self.chunksize = chunksize  # 块大小
        self.header = bool(header)  # 是否包含头部

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))  # 压缩图像
        if not self.header:
            assert buf.shape == self.shape  # 确保形状匹配
            assert buf.dtype == self.dtype  # 确保数据类型匹配
        return imagecodecs.zfp_encode(
            buf,
            level=self.level,
            mode=self.mode,
            execution=self.execution,
            header=self.header,
            numthreads=self.numthreads,
            chunksize=self.chunksize,
        )  # 编码并返回结果

    def decode(self, buf, out=None):
        if self.header:
            return imagecodecs.zfp_decode(buf, out=out)  # 解码并返回结果
        return imagecodecs.zfp_decode(
            buf,
            shape=self.shape,
            dtype=numpy.dtype(self.dtype),
            strides=self.strides,
            numthreads=self.numthreads,
            out=out,
        )  # 解码并返回结果

class Zlib(Codec):
    """Zlib编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_zlib'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        return imagecodecs.zlib_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.zlib_decode(buf, out=_flat(out))  # 解码并返回结果

class Zlibng(Codec):
    """Zlibng编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_zlibng'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        return imagecodecs.zlibng_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.zlibng_decode(buf, out=_flat(out))  # 解码并返回结果

class Zopfli(Codec):
    """Zopfli编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_zopfli'

    def encode(self, buf):
        return imagecodecs.zopfli_encode(buf)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.zopfli_decode(buf, out=_flat(out))  # 解码并返回结果

class Zstd(Codec):
    """ZStandard编解码器用于numcodecs。"""

    codec_id = 'imagecodecs_zstd'

    def __init__(self, level=None):
        self.level = level  # 编码等级

    def encode(self, buf):
        return imagecodecs.zstd_encode(buf, level=self.level)  # 编码并返回结果

    def decode(self, buf, out=None):
        return imagecodecs.zstd_decode(buf, out=_flat(out))  # 解码并返回结果

def _flat(out):
    """如果可能，将numpy数组返回为字节的连续视图。"""
    if out is None:
        return None
    view = memoryview(out)  # 创建内存视图
    if view.readonly or not view.contiguous:
        return None
    return view.cast('B')  # 转换为字节视图

def register_codecs(codecs=None, force=False, verbose=True):
    """使用numcodecs注册此模块中的编解码器。"""
    for name, cls in globals().items():
        if not hasattr(cls, 'codec_id') or name == 'Codec':
            continue
        if codecs is not None and cls.codec_id not in codecs:
            continue
        try:
            try:
                get_codec({'id': cls.codec_id})  # 获取编解码器
            except TypeError:
                # 已注册，但失败
                pass
        except ValueError:
            # 尚未注册
            pass
        else:
            if not force:
                if verbose:
                    log_warning(
                        f'numcodec {cls.codec_id!r} 已注册'
                    )
                continue
            if verbose:
                log_warning(f'替换已注册的numcodec {cls.codec_id!r}')
        register_codec(cls)  # 注册编解码器

def log_warning(msg, *args, **kwargs):
    """记录警告级别的消息。"""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)  # 记录警告消息
