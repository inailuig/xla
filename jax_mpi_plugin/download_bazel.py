#!/usr/bin/python
#
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Helper script for building JAX's libjax easily.


import argparse
import collections
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import textwrap
import urllib

# pylint: disable=g-import-not-at-top
if hasattr(urllib, "urlretrieve"):
  urlretrieve = urllib.urlretrieve
else:
  import urllib.request
  urlretrieve = urllib.request.urlretrieve

if hasattr(shutil, "which"):
  which = shutil.which
else:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


def is_windows():
  return sys.platform.startswith("win32")


def shell(cmd):
  try:
    output = subprocess.check_output(cmd)
  except subprocess.CalledProcessError as e:
    print(e.output)
    raise
  return output.decode("UTF-8").strip()


# Python

def get_python_bin_path(python_bin_path_flag):
  """Returns the path to the Python interpreter to use."""
  path = python_bin_path_flag or sys.executable
  return path.replace(os.sep, "/")


def get_python_version(python_bin_path):
  version_output = shell(
    [python_bin_path, "-c",
     ("import sys; print(\"{}.{}\".format(sys.version_info[0], "
      "sys.version_info[1]))")])
  major, minor = map(int, version_output.split("."))
  return major, minor

def check_python_version(python_version):
  if python_version < (3, 9):
    print("ERROR: JAX requires Python 3.9 or newer, found ", python_version)
    sys.exit(-1)


def check_numpy_version(python_bin_path):
  version = shell(
      [python_bin_path, "-c", "import numpy as np; print(np.__version__)"])
  numpy_version = tuple(map(int, version.split(".")[:2]))
  if numpy_version < (1, 22):
    print("ERROR: JAX requires NumPy 1.22 or newer, found " + version + ".")
    sys.exit(-1)
  return version

# Bazel

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/6.1.2/"
BazelPackage = collections.namedtuple("BazelPackage",
                                      ["base_uri", "file", "sha256"])
bazel_packages = {
    ("Linux", "x86_64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.1.2-linux-x86_64",
            sha256=
            "e89747d63443e225b140d7d37ded952dacea73aaed896bca01ccd745827c6289"),
    ("Linux", "aarch64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.1.2-linux-arm64",
            sha256=
            "1c9b249e315601c3703c41668a1204a8fdf0eba7f0f2b7fc38253bad1d1969c7"),
    ("Darwin", "x86_64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.1.2-darwin-x86_64",
            sha256=
            "22d4b605ce6a7aad92d4f387458cc68de9907a2efa08f9b8bda244c2b6010561"),
    ("Darwin", "arm64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.1.2-darwin-arm64",
            sha256=
            "30cdf85af055ca8fdab7de592b1bd64f940955e3f63ed5c503c4e93d0112bd9d"),
    ("Windows", "AMD64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.1.2-windows-x86_64.exe",
            sha256=
            "47e7f65a3bfa882910f76e2107b4298b28ace33681bd0279e25a8f91551913c0"),
}


def download_and_verify_bazel():
  """Downloads a bazel binary from Github, verifying its SHA256 hash."""
  package = bazel_packages.get((platform.system(), platform.machine()))
  if package is None:
    return None

  if not os.access(package.file, os.X_OK):
    uri = (package.base_uri or BAZEL_BASE_URI) + package.file
    sys.stdout.write(f"Downloading bazel from: {uri}\n")

    def progress(block_count, block_size, total_size):
      if total_size <= 0:
        total_size = 170**6
      progress = (block_count * block_size) / total_size
      num_chars = 40
      progress_chars = int(num_chars * progress)
      sys.stdout.write("{} [{}{}] {}%\r".format(
          package.file, "#" * progress_chars,
          "." * (num_chars - progress_chars), int(progress * 100.0)))

    tmp_path, _ = urlretrieve(uri, None,
                              progress if sys.stdout.isatty() else None)
    sys.stdout.write("\n")

    # Verify that the downloaded Bazel binary has the expected SHA256.
    with open(tmp_path, "rb") as downloaded_file:
      contents = downloaded_file.read()

    digest = hashlib.sha256(contents).hexdigest()
    if digest != package.sha256:
      print(
          "Checksum mismatch for downloaded bazel binary (expected {}; got {})."
          .format(package.sha256, digest))
      sys.exit(-1)

    # Write the file as the bazel file name.
    with open(package.file, "wb") as out_file:
      out_file.write(contents)

    # Mark the file as executable.
    st = os.stat(package.file)
    os.chmod(package.file,
             st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  return os.path.join(".", package.file)


if __name__ == "__main__":
  download_and_verify_bazel()
