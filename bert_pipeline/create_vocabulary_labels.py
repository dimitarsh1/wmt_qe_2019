# -*- coding: utf-8 -*-

''' reads a text file and exports unique tokens separated by space.
'''
import argparse
import codecs
import os
import sys

def main():
    ''' main function '''
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Extract vocabulary.')
    parser.add_argument('-d', '--datafile', required=True, help='the data to extract vocabulary.')

    args = parser.parse_args()
    
    # initialize a vocabulary as a set (only one occurrence of an element)
    data_vocabulary = {}
    if os.path.exists(args.datafile):
        with codecs.open(args.datafile, 'r', 'utf8') as fh:
            for line in fh:
                # do the union to add more elements to the vocabulary
                for token in line.split():
                    if token not in data_vocabulary:
                        data_vocabulary[token] = 1
                    else:
                        data_vocabulary[token] += 1

        # print the vocabulary to stdout.
        print("\n".join(sorted(data_vocabulary, key=data_vocabulary.get, reverse=True)))
    else:
        # if file doesn't exist exit and print a message to stderr`
        print("ERROR: Path not found: ", args.datafile, file=sys.stderr)
        exit(1)  

if __name__ == "__main__":
    main()
