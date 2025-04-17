load("@rules_python//python:defs.bzl", "py_binary")
load("@my_project_pip_hub//:requirements.bzl", "requirement")

py_binary(
    name = "cat_identifier",
    main = "projects/cat_identifier/detector.py",
    srcs = ["projects/cat_identifier/detector.py"],
    deps = [
        requirement("opencv-python"),
        requirement("ultralytics"),
        # requirement("tensorflow"),
    ],
)

# py_binary(
#     name = "tessie",
#     main = "projects/tessie_reader.py",
#     srcs = ["projects/tessie_reader.py"],
#     deps = [
#         requirement("pandas"),
#         requirement("geopandas"),
#         requirement("folium"),
#     ],
# )

# py_binary(
#     name = "meow",
#     main = "projects/cat_sound_finder.py",
#     srcs = ["projects/cat_sound_finder.py"],
#     deps = [
#         requirement("librosa"),
#         requirement("moviepy"),
#         requirement("imageio_ffmpeg"),
#         requirement("numpy"),
#     ],
# )

py_binary(
    name = "my_cli",
    main = "projects/cli/cli_run.py",
    srcs = [
        "projects/cli/cli_run.py",
        "projects/cli/person_pb2.py",
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

py_binary(
    name = "cli-run",
    main = "projects/scenario_tag_bulk_cli/scenario_tagging.py",
    srcs = [
        "projects/scenario_tag_bulk_cli/scenario_tagging.py",
        "projects/scenario_tag_bulk_cli/scenario_tags_pb2.py",
        ],
    deps = [
        requirement("click"),
        requirement("protobuf"),
    ],
)

py_binary(
    name = "proto_cli_parser",
    main = "projects/click_custom_proto_parser/proto_cli_parser.py",
    srcs = [
        "projects/click_custom_proto_parser/proto_cli_parser.py",
        "projects/click_custom_proto_parser/joey_pb2.py",
        ],
    deps = [
        requirement("click"),
        requirement("protobuf"),
    ],
)

py_binary(
    name = "photo_resizer",
    main = "projects/photo_resizer/resize.py",
    srcs = [
        "projects/photo_resizer/resize.py",
        ],
    deps = [

    ],
)

py_binary(
    name = "backup_analyzer",
    main = "projects/backup_analyzer/backup_analyzer.py",
    srcs = ["projects/backup_analyzer/backup_analyzer.py"],
    deps = [
        requirement("tqdm"),
        requirement("psutil"),
    ],
)