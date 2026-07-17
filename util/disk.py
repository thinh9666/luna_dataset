import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import MODE_BINARY
from io import BytesIO
from pathlib import Path
from util.logconf import logging
import pickle
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

CACHE_ROOT = (
    Path(__file__).resolve().parent
    / "data-unversioned"
    / "cache"
)
class GzipDisk(Disk):#nén gzip => gọi Disk.store() để lưu
    def store(self, value, read, key=None):
        #value: dữ liệu cần lưu cache
        #cho biết value có phải file-like object cần gọi .read() không.
        #key khóa cache, đoạn này ko dùng
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :return: (size, mode, filename, value) tuple for Cache table
        """
        if read:# nếu là file thì phải read
            value = value.read()# đọc thành bytes
            read= False
        '''
        buf = io.BytesIO()
        with gzip.GzipFile(
            fileobj=buf,
            mode="wb",
            compresslevel=1,
        ) as gz_file:
            for offset in range(0, len(value), 2**30):
                gz_file.write(value[offset:offset + 2**30])

        value = buf.getvalue()
        '''# vì ghi xuống ram nên xài compress cho nhanh
            #sau khi đọc xong, kiểm tra dữ liệu xem có phải bytes không
         # Chuyển cả tuple, NumPy array, namedtuple... thành bytes.
        serialized = pickle.dumps(value,protocol = pickle.HIGHEST_PROTOCOL)#chuyển value thành bytes
        if isinstance(value,serialized):# xài
            value = gzip.compress(
                serialized, compresslevel = 1
            )
        return super().store(
            value,read, key=key
        )

    def fetch(self, mode, filename, value, read):
        #mode cho biết giá trị được lưu dưới dạng nào, trường hợp này là byte
        #file name là Là đường dẫn hoặc tên file chứa dữ liệu cache
        #value là Là giá trị được đọc từ database của cache.
        ##read true hoặc false  là trả về file object hoặc nội dung dữ liệu 
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        compressed = super(GzipDisk, self).fetch(mode, filename, value, read = False)
        '''
        if mode == MODE_BINARY:
            str_io = BytesIO(value)# file giả chứ dữ liệu nén
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)#đọc và giải nén
            read_csio = BytesIO()#

            while True:
                uncompressed_data = gz_file.read(2**30) #từng khối đã giải nén
                if uncompressed_data:
                    read_csio.write(uncompressed_data)#gom các khối lại
                else:
                    break

            value = read_csio.getvalue()
        '''
        serialized = gzip.decompress(compressed)
        return pickle.loads(serialized)#chuyển bytes thành lại giá trị thật

def getCache(scope_str):
    #scope_str Là tên phạm vi cache, dùng để tạo thư mục riêng cho từng loại dữ liệu.
    #ví dụ raw_cache = getCache('part2ch12_raw')
    return FanoutCache('data-unversioned/cache/' + scope_str,#'data-unversioned/cache/part2ch12_raw'
                       disk=GzipDisk,#Chỉ định lớp chịu trách nhiệm chuyển dữ liệu Python thành dữ liệu có thể lưu trên ổ đĩa
                       shards=8,#chia cache thành 64 shards, giảm xung đột khi nhiều tiến trình cùng đọc/ghi
                       #Khi lưu một key, FanoutCache sẽ dựa vào hash của key để quyết định key đó thuộc shard nào.
                       timeout=5,#Thời gian tối đa, tính bằng giây, mà cache chờ khi database đang bị khóa.
                       size_limit=50 * 1024**3,#Đây là giới hạn dung lượng cache, tính bằng byte.
                       # disk_min_file_size=2**20,
                       )

# def disk_cache(base_path, memsize=2):
#     def disk_cache_decorator(f):
#         @functools.wraps(f)
#         def wrapper(*args, **kwargs):
#             args_str = repr(args) + repr(sorted(kwargs.items()))
#             file_str = hashlib.md5(args_str.encode('utf8')).hexdigest()
#
#             cache_path = os.path.join(base_path, f.__name__, file_str + '.pkl.gz')
#
#             if not os.path.exists(os.path.dirname(cache_path)):
#                 os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#
#             if os.path.exists(cache_path):
#                 return pickle_loadgz(cache_path)
#             else:
#                 ret = f(*args, **kwargs)
#                 pickle_dumpgz(cache_path, ret)
#                 return ret
#
#         return wrapper
#
#     return disk_cache_decorator
#
#
# def pickle_dumpgz(file_path, obj):
#     log.debug("Writing {}".format(file_path))
#     with open(file_path, 'wb') as file_obj:
#         with gzip.GzipFile(mode='wb', compresslevel=1, fileobj=file_obj) as gz_file:
#             pickle.dump(obj, gz_file, pickle.HIGHEST_PROTOCOL)
#
#
# def pickle_loadgz(file_path):
#     log.debug("Reading {}".format(file_path))
#     with open(file_path, 'rb') as file_obj:
#         with gzip.GzipFile(mode='rb', fileobj=file_obj) as gz_file:
#             return pickle.load(gz_file)
#
#
# def dtpath(dt=None):
#     if dt is None:
#         dt = datetime.datetime.now()
#
#     return str(dt).rsplit('.', 1)[0].replace(' ', '--').replace(':', '.')
#
#
# def safepath(s):
#     s = s.replace(' ', '_')
#     return re.sub('[^A-Za-z0-9_.-]', '', s)
