import sys
import click
import csv
import json
from google.protobuf.json_format import ParseDict, ParseError
import cli.person_pb2 as person_pb2

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input', type=click.File('r'), default=sys.stdin)
def load_csv(input):
    # Read the CSV file from stdin
    input_data = input.read()
    csv_reader = csv.DictReader(input_data.splitlines())


    
    # Convert CSV data to a list of dictionaries
    records = []
    for row in csv_reader:
        print(f"Row: {row}")  # For debugging, print each row to check the content
        record = {}
        for key, value in row.items():
            if key and value:  # Only add fields with non-None keys and values
                if key == 'hobbies' and value.strip():
                    record[key] = [hobby.strip() for hobby in value.split(',')]  # Handle hobbies field correctly
                else:
                    keys = key.split('.')
                    sub_dict = record
                    for k in keys[:-1]:
                        if k not in sub_dict:
                            sub_dict[k] = {}
                        sub_dict = sub_dict[k]
                    sub_dict[keys[-1]] = value
        records.append(record)
    
    # Print JSON data (for debugging purposes)
    print("JSON data:")
    print(json.dumps(records, indent=2))
    
    # Convert JSON to protobuf
    for record in records:
        try:
            person_pb = ParseDict(record, person_pb2.Person())
            print("Protobuf message:")
            print(person_pb)
        except ParseError as e:
            click.echo(f"Failed to parse JSON to Protobuf: {e}", err=True)
            sys.exit(1)

if __name__ == '__main__':
    cli()
