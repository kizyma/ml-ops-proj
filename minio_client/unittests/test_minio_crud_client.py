import os
import unittest
from minio_client.minio_crud_client import MinioCRUDClient


class TestMinioCRUDClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        cls.access_key = os.getenv("MINIO_ACCESS_KEY", "minio")
        cls.secret_key = os.getenv("MINIO_SECRET_KEY", "minio123")
        cls.client = MinioCRUDClient(cls.endpoint, cls.access_key, cls.secret_key)
        cls.bucket_name = "testbucket"
        cls.client.create_bucket(cls.bucket_name)

    def setUp(self):
        # Set up the environment before each test
        self.test_file_name = "test.txt"
        self.test_file_content = "This is a test file."
        with open(self.test_file_name, "w") as f:
            f.write(self.test_file_content)
        self.client.upload_file(
            self.bucket_name, self.test_file_name, self.test_file_name
        )

    def test_upload_file(self):
        uploaded = self.client.upload_file(
            self.bucket_name, self.test_file_name, self.test_file_name
        )
        self.assertTrue(uploaded)

    def test_download_file(self):
        downloaded = self.client.download_file(
            self.bucket_name, self.test_file_name, "downloaded_test.txt"
        )
        self.assertTrue(downloaded)
        # Check if the downloaded file has the correct content
        with open("downloaded_test.txt", "r") as f:
            content = f.read()
        self.assertEqual(content, self.test_file_content)

    def test_delete_file(self):
        deleted = self.client.delete_file(self.bucket_name, self.test_file_name)
        self.assertTrue(deleted)

    def tearDown(self):
        # Clean up the environment after each test
        if os.path.exists(self.test_file_name):
            os.remove(self.test_file_name)
        if os.path.exists("downloaded_test.txt"):
            os.remove("downloaded_test.txt")

    @classmethod
    def tearDownClass(cls):
        # Clean up by removing the bucket and its contents after all tests
        objects_to_delete = cls.client.list_objects(cls.bucket_name)
        for obj in objects_to_delete:
            cls.client.delete_file(cls.bucket_name, obj.object_name)
        if cls.client.client.bucket_exists(cls.bucket_name):
            cls.client.client.remove_bucket(cls.bucket_name)


if __name__ == "__main__":
    unittest.main()
