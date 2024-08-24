import click
from google.protobuf.message import Message
from projects.click_custom_proto_parser import joey_pb2
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING,  # Set the logging level
                    format='%(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)  # Create a logger instance

@click.group()
def cli():
    pass

class ProtoMessageParser(click.ParamType):
    name = "proto_message"

    def __init__(self, proto_message_class):
        super().__init__()
        self.proto_message_class = proto_message_class

    def convert(self, value, param, ctx):
        proto_message = self.proto_message_class()

        for arg in value:
            logger.debug(f"Processing argument: {arg}")

            try:
                field_path, field_value = arg.split("=", 1)
            except ValueError:
                self.fail(f"Invalid argument format: {arg}. Use field.subfield=value", param, ctx)

            logger.debug(f"Parsed field path: {field_path}, value: {field_value}")

            fields = field_path.split('.')
            current_field = proto_message

            try:
                for field in fields[:-1]:
                    logger.debug(f"Navigating to field: {field}")
                    current_field = getattr(current_field, field)
                setattr(current_field, fields[-1], field_value)
                logger.debug(f"Set {fields[-1]} to {field_value}")
            except AttributeError as e:
                self.fail(f"Invalid field path: {field_path}. Error: {e}", param, ctx)

        return proto_message

@click.command()
@click.argument('req_1')
@click.argument('req_2')
@click.option('--file', 'file', type=click.File('r'), default=sys.stdin, required=False, help="Path to the input file or stdin")
@click.argument('addtional_fields', nargs=-1)
def to_proto(req_1, req_2, file, addtional_fields):
    all_args = ["req_1="+req_1, "req_2="+req_2] + list(addtional_fields)

    proto_parser = ProtoMessageParser(proto_message_class=joey_pb2.YourProtoMessage)
    proto_message = proto_parser.convert(all_args, None, None)
    
    click.echo(f"Parsed Protobuf Message: {proto_message}")

cli.add_command(to_proto)

if __name__ == '__main__':
    cli()
