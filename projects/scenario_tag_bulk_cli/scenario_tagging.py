import sys
import os
import click
import logging
import csv
import json
from google.protobuf.json_format import ParseDict, ParseError
import projects.scenario_tag_bulk_cli.scenario_tags_pb2 as scenario_tags_pb2

# Set up logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)  # Create a logger instance

@click.group()
def cli():
    pass


def _process_csv(input):
    csv_reader = csv.DictReader(input.splitlines())
    records = []
    for row in csv_reader:
        record = {}
        for key, value in row.items():
            if key and value:
                if key == 'tags' and value.strip():
                    record[key] = [tag.strip() for tag in value.split(',')]
                else:
                    keys = key.split('.')
                    sub_dict = record
                    for k in keys[:-1]:
                        if k not in sub_dict:
                            sub_dict[k] = {}
                        sub_dict = sub_dict[k]
                    sub_dict[keys[-1]] = value
        records.append(record)
    return records

@cli.command()
@click.argument('input', type=click.File('r'), default=sys.stdin)
def add_tags(input):
    # Read the CSV file from stdin
    protos_list = []
    input_data = input.read()
    records = _process_csv(input_data)
    
    # Print JSON data (for debugging purposes)
    logger.info(json.dumps(records, indent=2))
    
    # Convert dicts to protobuf
    for record in records:
        try:
            scenario_tag_pb = ParseDict(record, scenario_tags_pb2.ScenarioTags())
            protos_list.append(scenario_tag_pb)
            logger.info("Protobuf message:")
            logger.info(scenario_tag_pb)
        except ParseError as e:
            click.echo(f"Failed to parse JSON to Protobuf: {e}", err=True)
            sys.exit(os.EX_DATAERR)

    [click.echo(proto.scenario_id) for proto in protos_list]

if __name__ == '__main__':
    cli()
