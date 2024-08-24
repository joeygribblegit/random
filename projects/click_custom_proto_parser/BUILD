load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_binary(
    name = "cli-run",
    main = "proto_cli_parser.py",
    srcs = [
        "proto_cli_parser.py",
        "joey_pb2.py",
        ],
    deps = [
        requirement("click"),
        requirement("protobuf"),
    ],
)

