#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Dish classification

.. moduleauthor:: Paul Karadimas <paulkar@gmail.com>

Description(given):
  The problem: 
    Given a sample of salads and sandwiches from a menu, 
    programmically identify a Tuna salad sandwich as a "sandwich"
    and not a "salad".

    Attached is a csv of menu item name, description, and price. 
    It only has 15 rows. Half the rows are salads. 
    Half the rows are sandwiches. 
    The last one, row 15, is the Tuna fish salad.  
    Your solution should also work for something like an 
    "egg salad" sandwich or a "chicken salad" sandwich.

    It’s a very small sample set intentionally. 
    Don’t spend too much time on this. Remember, 2 hours max.

Python Orange library lessons learned:
  * The tab delimited input format is finicky, or maybe I just glanced too
    far over the details of it. I didn't try C4.5

    There must be 3 header lines though the documentation doesn't really
    give that impression, the forums did.

    If your base file is menu_sample.csv and you make menu_sample.csv.tab,
    the library tries to open the csv and not the tab. I wasn't even aware
    that csv was an excepted format from the documentation.
    I spent more time trying to figure this out than anything else I think.

  * Documentation is ok but not great, they could use a proof reader and maybe
    a spell checker

  * I didn't have a chance to use the plotting libraries available but
    they may be useful. The spec asked for programmatically solving the
    problem so I left those out.

  * I didn't read up enough and am not sure what the difference is between
    import orange, import Orange

Pre-Processing:
  The input csv file is converted to an orange excepted tab delimited format
  of word vectors and the number of occurrences.

  name  word1 word2 word3

  name1 1   4   2
  name2 3   1   2
  name3 5   1   3

Solutions:
  K-means clustering:
    Since the number of clusters is known here as k=2(salad, sandwich) 
    this is simple and works in the alotted time. 
    Also since no training set was given I'm only referring to
    unsupervised algorithms.

    From memory this algorithm does converge but may take exponential time
    so likely wouldn't be a great choice for a larger dataset. There is
    Hierarchical Clustering but I think the performance is substantially 
    worse and it requires more work to find clusters.

    The selection of centroids is random and since the dataset is so small
    this works, but again larger datasets and an unknown number of clusters
    makes this not ideal.

    It's not 100% accurate but more data may normalize that along with 
    better filtering of words. 

  Alternatives:
    Feature extraction algorithms:
      ICA would might be interesting, treating each row as a 
      multivariate signal to help find patterns. 
      Non-negative matrix factorization falls into the category as well.

    Supervised learning algorithms:
      A plethora of choices here too involved to fit in the alotted time.
      Kernel methods come to mind with support vector machines.
      

Conclusion:
  * Given more time I'd probably look at using more advanced 
    Blind Source Separation algorithms like Independent Component Analysis 
    for unsurpervised learning.

Running:
  Requirements:
    * Orange machine learning library nightly build
      http://orange.biolab.si/nightly_builds.html

    * Tested with Python 2.6.5 on Mac OSX 10.7 using the Orange dmg
      version of the interpreter which by default if installed is in:
      /Applications/Orange.app/Contents/MacOS/python
"""

import os
import re
import csv
import sys
import logging
from matplotlib import pyplot as plt
logging.getLogger().setLevel(logging.DEBUG)

try:
  import Orange
  import OWKMeans
except ImportError:
  logging.error("""Please install the Orange module from """
      """http://orange.biolab.si/nightly_builds.html""")
  exit()

def usage():
  print \
    """
    Usage: python foodie.py menu_sample.csv

    """

class CSVImportError(Exception):
  """ Generic file input error.

  """
  pass
  
class Utils():
  """ Random text processing utilities.

  """
  @classmethod
  def get_word_counts(cls, row):
    """ This assumes the row given is in the format
    [name, description, price] and returns a dict
    of words and their counts for the row

    Args:
      * `row` (list): Assumes

    Returns:
      ``dict`` {word1:count, word2:count}

    >>> row = ['name', 'some description', '1.11']
    >>> Utils.get_word_counts(row)
    {'1.11': 1, 'some': 1, 'name': 1, 'description': 1}

    """
    words = [re.sub('\W+', '', wrd.lower()) for wrd in 
        row[0].split() + row[1].split() if len(wrd) > 1]
    words += [r for r in row[2].split() if r]

    word_counts = {}
    for word in words:
      word_counts.setdefault(word, 0)
      word_counts[word] += 1
    return word_counts

  @classmethod
  def convert_csv_to_tab_word_vectors(
      cls, csv_reader, filename):
    """ Convert a comma separated file to tab separation.
    [name, description, price] is assumed
    This is the default format for the Orange library

    Args:
      * `csv_reader` (csv.reader): csv file reader
      * `filename` (str, unicode): name of file input

    Returns:
      None

    >>> filename = menu_sample.csv
    >>> csv_reader = csv.reader(open(filename, 'rU'), delimiter=',')
    >>> Utils.convert_csv_to_tab_word_vectors(csv_reader, filename)
 
    """
    if not csv_reader:
      raise CSVImportError

    logging.info('Converting ' + filename + ' to tab word vectors')
    columns = [cell.strip() for cell in csv_reader.next()]

    # Word counts for the row
    word_counts = {}
    # Number of rows each word appears in, used to filter
    row_counts = {}
    for row in csv_reader:
      word_counts[row[0]] = cls.get_word_counts(row)
      for word, count in word_counts[row[0]].iteritems():
        row_counts.setdefault(word, 0)
        if count >= 1:
          row_counts[word] += 1

    words = []
    # Here you might experiment with filtering out words that
    # appear too frequently. 
    # In such a small dataset all the words are included
    # Something like count/total words and using what 
    # falls between percentages
    # Or use stop words
    for word, count in row_counts.iteritems():
      words.append(word)

    file_name_root, file_extension = os.path.splitext(filename)
    tab_file = open(file_name_root + '.tab', 'wb')
    tab_writer = csv.writer(tab_file, delimiter='\t')
    tab_writer.writerow(['name'] + words)
    # Discrete vs continuous line
    tab_writer.writerow(['string'] + ['c'] * len(words))
    # Blank line needed
    tab_file.write('\n')
    for name, word_counts in word_counts.iteritems():
      row = [name]
      for word in words:
        row.append(word_counts.get(word, 0))

      tab_writer.writerow(row)

    tab_file.close()


if __name__ == "__main__":

  try:
    filename = sys.argv[1]
  except (IOError, IndexError):
    usage()
    raise CSVImportError

  logging.info('Loading file ' + filename + '...')
  csv_reader = csv.reader(open(filename, 'rU'), delimiter=',')
  Utils.convert_csv_to_tab_word_vectors(csv_reader, filename)

  # Perform k-means clustering k=2 using Orange
  logging.info('Calculating k-means clustering...')
  file_name_root, file_extension = os.path.splitext(filename)

  # 1=sandwich=s 0=salad=l
  # Hardcoded to get a performance metric
  # you may have to swap 1 and 0
  s = 0
  l = 1
  actuals_dict = {
    'Slenderella Salad': l,
    'La Conga Delight': s,
    'Chicken Caesar Salad': l,
    'Open Face Chunk White Skipjack Tunafish Salad': s,
    'BLT Club': s,
    'Royal Burger Delight': s,
    'Small Tossed Salad': l,
    'Refreshing Greek Salad': l,
    'Hollywood Salad Bowl': l,
    'Filet Of Sole': s,
    'Fresh Garden Spinach Salad': l,
    'Fresh Breast of Chicken': s,
    'Caesar Salad': l,
    "Mitchell's Club House": s,
  }

  ###########################################################################
  # Plucked from the orange examples
  ###########################################################################
  def callback(km):
    """ Uncomment to see iterations

    """
    pass
    #print "Iteration: %d, changes: %d, score: %.4f" % (
    #    km.iteration, km.nchanges, km.score)
  table = Orange.data.Table(file_name_root + '.tab')
  actuals_values = []
  for row in table:
    actuals_values.append(actuals_dict[unicode(row[0])])

  # Iterate distance metrics and get actual and calculated results
  most_accurate = 0.0
  final_choice = ''
  for measure in OWKMeans.OWKMeans.distanceMeasures:
    km = Orange.clustering.kmeans.Clustering(
      table, 2, minscorechange=0, inner_callback=callback, 
      initialization=Orange.clustering.kmeans.init_random,
      distance=measure[1],
    )
    print '#' * 80
    print 'K-means clustering k=2 random centroid initialization'
    print measure[0]
    for index, value in enumerate(km.clusters):
      print '\t', km.clusters[index], table[index][0]

    correct = 0.0
    print
    print 'Correct values: ' 
    for index, val in enumerate(actuals_values):
      if km.clusters[index] == val:
        print '\t', km.clusters[index], table[index][0]
        correct += 1
    calc = float(correct)/float(len(actuals_values))

    print 'Calculated ', km.clusters
    print 'Actual   ', actuals_values
    print 'Accuracy: ', calc
    print '#' * 80
    if calc > most_accurate:
      most_accurate = calc
      final_choice = 'Best result: %s %s correct %s' % (
          measure[0], int(correct), calc)
  print '*' * 80
  print final_choice
  print '*' * 80
  logging.info('Note: You may need to switch between 1 and 0 for ' + \
      'sandwich and salad line 256. This is just to get an accuracy ' \
      'number and not to cluster the data')
