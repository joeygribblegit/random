load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_library(
    name = "my_deps",
    srcs = [],
    deps = [
        requirement("pandas"),
        requirement("geopandas"),
        requirement("folium"),
    ],
)

py_binary(
    name = "tessie",
    main = "reader.py",
    srcs = ["reader.py"],
    deps = [":my_deps"],
)

py_binary(
    name = "meow",
    main = "cat_sound_finder.py",
    srcs = ["cat_sound_finder.py"],
    deps = [
        requirement("librosa"),
        requirement("moviepy"),
        requirement("imageio_ffmpeg"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "my_cli",
    main = "cli/cli_run.py",
    srcs = [
        "cli/cli_run.py",
        "cli/person_pb2.py",
        ],
    deps = [
        requirement("click"),
        requirement("protobuf"),
    ],
)

py_binary(
    name = "file_search",
    main = "projects/file_search/file_search.py",
    srcs = [
        "projects/file_search/file_search.py",
        ],
    deps = [
        requirement("pytsk3"),
    ],
)

py_binary(
    name = "metadata_updater",
    main = "projects/metadata_updater/updater.py",
    srcs = [
        "projects/metadata_updater/updater.py",
        ],
    deps = [
        requirement("piexif"),
        requirement("Pillow"),
    ],
)

py_binary(
    name = "time_reverser",
    main = "projects/metadata_updater/reverse_order.py",
    srcs = [
        "projects/metadata_updater/reverse_order.py",
        ],
    deps = [
        requirement("piexif"),
        requirement("Pillow"),
    ],
)