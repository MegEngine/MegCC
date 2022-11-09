#!/usr/bin/env python3
# -*-coding=utf-8-*-

import boto3
import datetime
import hashlib
import json
import os
import random
import redis
import string
import subprocess
import time
import yaml
import argparse
from pathlib import Path
from korok import bundle,task

def parse_config(config_file: str) -> dict:
    with open(config_file,'r') as f:
        data = yaml.safe_load(f)
        return data

def generate_random_id() -> str:
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))

# compute the hash of file
def hash_path(path):
    files_hash = {}

    def hashy(path):
        assert os.path.exists(path),'cannot find path {}'.format(path)
        if os.path.isfile(path):
            ho = hashlib.sha256()
            with open(path, "rb") as fp:
                for chunk in iter(lambda: fp.read(1024 ** 2), b""):
                    ho.update(chunk)
            files_hash[path] = ho.hexdigest()
        elif os.path.isdir(path):
            for r, d, fs in os.walk(path):
                for f in fs:
                    hashy(os.path.join(r, f))

    hashy(path)
    return files_hash


TOOL_INPUT = [
    "bazel_remote_test.py",
]

# redis result wrapper
class RedisResultCache():
    _conn = None  # type: redis.StrictRedis
    _ttl = None

    def __init__(self, redis_url: str, ttl: datetime.date):
        self._conn = redis.from_url(redis_url)
        self._ttl = ttl

    def get(self, key: str):
        self._conn.expire(key, self._ttl)
        return self._conn.get(key)

    def put(self, key: str, result: bool):
        return self._conn.set(key, result, ex=self._ttl)

class BaseConfig():
    def __init__(self):
        self.korok_server=os.environ.get('KOROK_SERVER')
        self.korok_token=os.environ.get('KOROK_TOKEN')
        self.korok_group=os.environ.get('KOROK_GROUP')
        assert self.korok_token, "korok token cannot be empty!"
        self.oss_access_key_id=os.environ.get('OSS_ACCESS_KEY_ID')
        self.oss_secret_access_key=os.environ.get('OSS_SECRET_ACCESS_KEY')
        if not self.oss_access_key_id:
            session = boto3.Session()
            credentials = session.get_credentials()
            self.oss_access_key_id=credentials.access_key
            self.oss_secret_access_key=credentials.secret_key
        self.oss_internal_host = os.environ.get('OSS_ENDPOINT')
        self.oss_external_host=os.environ.get('OSS_EXTERNAL_ENDPOINT')
        self.oss_bucket = os.environ.get('KOROK_BUCKET')
        redis_url = os.getenv("MGB_CI_CACHE_REDIS_URL")
        self.result_cache=None
        if redis_url:
            self.result_cache = RedisResultCache(redis_url, datetime.timedelta(weeks=1))

class Test():
    def __init__(self,config_file: str,name: str, id_hash: str,base_config: BaseConfig):
        config=parse_config(config_file)
        self.__id_hash=id_hash
        self.__name=name
        self.__target=config.get(name)
        assert self.__target,'cannot find target {0}'.format(name)
        settings=self.__target.get('settings')
        self.__use_bundle=None
        self.__device_tags=None
        self.__oss_enabled=None
        self.__bundle_id=None
        if settings:
            self.__use_bundle=settings.get('use-bundle')
            self.__device_tags=settings.get('device-tags')
            self.__oss_enabled=settings.get('oss-enabled')
        self.s3_client=None
    # get the result cache key, the key is affected with the following:
    # 1. test cmd 2. test envs 3. device tags 4. all input file 5. this script
    def compute_cache_key(self,metadata: list,files: list) -> str:
        files_hash=[]
        for file_path in files+TOOL_INPUT:
            print(file_path)
            file_hash = hash_path(file_path)
            assert len(file_hash), file_path
            files_hash.append(file_hash)
        blob = json.dumps(metadata + [files_hash], sort_keys=True)
        print("Cache key: {}".format(blob))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    def run_local(self,step:dict):
        build_envs=os.environ.copy()
        step_envs=step.get('envs')
        step_timeout=step.get('timeout')
        step_cmd=step.get('cmd')
        if step_envs:
            build_envs.update(step_envs)
        subprocess.run(args='&&'.join(step_cmd),shell=True,env=build_envs,timeout=step_timeout,check=True,stderr=subprocess.STDOUT)

    def run_remote(self,step_name: str,remote_workdir: str,step: dict):
        device_tags=step.get('device-tags')
        if not device_tags:
            device_tags=self.__device_tags
        assert device_tags,'device tags in step {} of target {} cannot be empty '.format(step_name,self.__name)
        if self.__use_bundle and not self.__bundle_id:
            bundle_client=bundle.BundleClient(address=base_config.korok_server, token=base_config.korok_token, group=base_config.korok_group)
            bundle_id=bundle_client.create_bundle(tags=device_tags,release_on_failure=True)
            print('Created bundle {}'.format(bundle_id))
            self.__bundle_id=bundle_id
        cmd='mkdir -pv {}'.format(remote_workdir)
        clean_command=[]
        upload_files=[]
        oss_enabled=step.get('oss-enabled')
        if not oss_enabled:
            oss_enabled=self.__oss_enabled
        input_files=step.get('input-files')
        if oss_enabled and input_files:
            assert base_config.oss_access_key_id, 'korok_oss_access_key_id is empty'
            assert base_config.oss_secret_access_key ,'korok_oss_secret_access_key is empty'
            oss_endpoint=base_config.oss_internal_host
            if not os.environ.get('CI'):
                oss_endpoint=base_config.oss_external_host
            s3_client = boto3.client(
                's3',
                endpoint_url=oss_endpoint,
                aws_access_key_id=base_config.oss_access_key_id,
                aws_secret_access_key=base_config.oss_secret_access_key,
                )
            self.__s3_client=s3_client
            oss_prepare_cmd=[
                'unset LD_PRELOAD',
                'rm -f ~/.aws/credentials',
                'export PATH=~/.local/bin:/usr/local/bin:$PATH',
                'aws configure set aws_access_key_id {}'.format(base_config.oss_access_key_id),
                'aws configure set aws_secret_access_key {}'.format(base_config.oss_secret_access_key),
                'aws --endpoint-url={} s3 cp --recursive s3://{}/ {}/'.format(base_config.oss_internal_host, os.path.join(base_config.oss_bucket,self.__id_hash,self.__name,step_name),remote_workdir),
            ]
            cmd='&&'.join([cmd]+oss_prepare_cmd)
            for file in input_files:
                self.__s3_client.upload_file(file, base_config.oss_bucket, os.path.join(self.__id_hash,self.__name,step_name,os.path.basename(file)))
            clean_command=['rm -rf {}'.format(remote_workdir)]
        elif not oss_enabled and input_files:
            # use korok upload file
            for file_path in input_files:
                upload_files.append((file_path,remote_workdir+'/'))
        cmd='&&'.join([cmd,'cd {}'.format(remote_workdir)]+step.get('cmd'))
        output_files=step.get('output-files')
        download_files=[]
        if oss_enabled and output_files:
            for file_name in output_files:
                cmd='&&'.join([cmd,'aws --endpoint-url={} s3 cp {} s3://{}'.format(base_config.oss_internal_host,file_name,os.path.join(base_config.oss_bucket,self.__id_hash,self.__name,step_name,file_name))])
        elif not oss_enabled and output_files:
            download_files.append((file_name,'./'))
        taskClient=task.TaskClient(address=base_config.korok_server, token=base_config.korok_token, group=base_config.korok_group)
        taskClient.create_task_with_log(tags=device_tags, command=[cmd],clean_command=clean_command,upload=upload_files, download=download_files, envs=step.get('envs'),bundle_id=self.__bundle_id,remove=True)
        if oss_enabled and output_files:
            for file_name in output_files:
                    pairs=os.path.split(file_name)
                    if pairs[0]:
                        try:
                            os.mkdir(pairs[0])
                        except FileExistsError:
                            print("Directory " , pairs[0]," already exists")
                    self.__s3_client.download_file(base_config.oss_bucket,os.path.join(self.__id_hash,self.__name,step_name,file_name),pairs[1])
                    if pairs[0]:
                        os.rename(pairs[1],file_name)
    def run(self):
        try:
            for step_name in self.__target:
                if step_name=='settings':
                    continue
                step=self.__target.get(step_name)
                cache_files=step.get('cache-files')
                if not cache_files:
                    cache_files=step.get('input-files')
                if cache_files and base_config.result_cache:
                    metadata=[step_name,step.get('cmd'),step.get('envs'),step.get('device-tags')]
                    cache_hash_key=self.compute_cache_key(metadata,cache_files)
                    if base_config.result_cache.get(cache_hash_key)=='true':
                        print("Hit result cache: successful")
                        continue
                print('Run {} in {}'.format(step_name, self.__name))
                start_time = time.time()
                type=step.get('type')
                if type=='remote':
                    remote_workdir='-'.join(['ci',self.__id_hash,self.__name,step_name])
                    self.run_remote(step_name,remote_workdir,step)
                else:
                    self.run_local(step)
                end_time = time.time()
                print('{} duration: {}s'.format(step_name, end_time-start_time))
        except BaseException as e:
            exit(e)
        finally:
            if self.__bundle_id:
                bundle_client=bundle.BundleClient(address=base_config.korok_server, token=base_config.korok_token, group=base_config.korok_group)
                bundle_client.delete_bundle(bundle_id=self.__bundle_id)

# get job ID
base_config=BaseConfig()
hash_id=os.getenv('CI_PIPELINE_ID')
if not hash_id:
    hash_id=generate_random_id()

def main():
    parser = argparse.ArgumentParser(description='Remote test with test name.')
    parser.add_argument('tests', nargs='+', help='All the test cast names.')
    args = parser.parse_args()

    # execute in project path
    project_path = str(Path(__file__).resolve().parent.parent.parent.parent)
    os.chdir(project_path)

    print(os.getcwd())

    test_start_time = time.time()
    for case_name in args.tests:
        testConfig=Test("./compiler/script/ci/remote_test_config.yaml",case_name,hash_id,base_config)
        testConfig.run()
    test_end_time = time.time()

    print('TOTAL DURATION: {}s'.format(test_end_time-test_start_time))

if __name__ == '__main__':
    main()
