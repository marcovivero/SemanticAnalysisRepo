"""
Import sample data for engine
"""

import predictionio
import argparse

def import_events(client, file):
    data  = open(file, 'r')
    count = 0
    print('Importing data...')
    for elem in data:
        count += 1
        elem = elem.rstrip('\n').split("\t")
        client.create_event(
            event = "phrases",
            entity_type = "source",
            entity_id = int(elem[0]), # use PhraseID
            properties= {
                "sentiment" : int(elem[3]),
                "phrase" : elem[2]
            }
        )
    print('Imported {0} events.'.format(str(count)))
    data.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import sample data for semantic analysis engine")
    parser.add_argument('--access_key', default='invald_access_key')
    parser.add_argument('--url', default="http://localhost:7070")
    parser.add_argument('--file', default="./data/train2.tsv")

    args = parser.parse_args()
    print(args)

    client = predictionio.EventClient(
        access_key=args.access_key,
        url=args.url,
        threads=5,
        qsize=500)
    import_events(client, args.file)





