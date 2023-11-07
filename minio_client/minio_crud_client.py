from minio import Minio
from minio.error import S3Error


class MinioCRUDClient:
    def __init__(self, endpoint, access_key, secret_key):
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=False
        )

    def create_bucket(self, bucket_name):
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            return True
        except S3Error as e:
            print("Error:", e)
            return False

    def upload_file(self, bucket_name, object_name, file_path):
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            return True
        except S3Error as e:
            print("Error:", e)
            return False

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            return True
        except S3Error as e:
            print("Error:", e)
            return False

    def delete_file(self, bucket_name, object_name):
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            print("Error:", e)
            return False

    def list_objects(self, bucket_name, recursive=True):
        try:
            return list(self.client.list_objects(bucket_name, recursive=recursive))
        except S3Error as e:
            print("Error:", e)
            return []
