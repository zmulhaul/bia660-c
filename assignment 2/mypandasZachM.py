import csv
from collections import OrderedDict
from dateutil.parser import parse
class DataFrame(object):

    @classmethod
    def from_csv(cls, file_path, delimiting_character=',', quote_character='"'):
        with open(file_path, 'rU') as infile:
            reader = csv.reader(infile, delimiter=delimiting_character, quotechar=quote_character)
            data = []

            for row in reader:
                data.append(row)

            return cls(list_of_lists=data)


    def __init__(self, list_of_lists, header=True):
        if header:
            self.header = list_of_lists[0]
            self.data = list_of_lists[1:]
        else:
            self.header = ['column' + str(index + 1) for index, column in enumerate(list_of_lists[0])]
            self.data = list_of_lists

        self.data = [[value.strip() for value in row] for row in self.data]


        self.data = [OrderedDict(zip(self.header, row)) for row in self.data]

        if len(self.header) != len(set(self.header)):
            raise Exception('There are duplicate column names in the header')




    def add_rows(self,listOfConvertedValues):
       listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]

       for row in self.data:
        return data.append(row)

        self.listOfConvertedValues = [OrderedDict(zip(self.listOfConvertedValues, row)) for row in self.data]

        if len(self.column_name) != len(self.row in self.data):
         raise Exception('Numbers of rows do not match number of columns')


    def add_column(self, listOfConvertedValues, column_name):
        listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]

        for row in self.column_name:
            return data.append(row)

        if len(self.listOfConvertedValues) != len(self.header):
            raise Exception('List of values does not equal number of rows in data frame')


    def mean(self, column_name):
        listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]
        realMean = sum(listOfConvertedValues)/len(self[column_name])
        return realMean

    def min(self, column_name):
        listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]
        realMin = min(listOfConvertedValues)
        return realMin

    def max(self, column_name):
        listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]
        realMax = max(listOfConvertedValues)
        return realMax

    def median(self, column_name):
        listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]
        realMedian = sorted(listOfConvertedValues)[len(listOfConvertedValues)//2]
        return realMedian

        import numpy
        assert df.median('Price') == numpy.median(df['Price'])

    def stDev(self, column_name):
        listOfConvertedValues = [convertToFloatOrDatetime(value) for value in self[column_name]]
        realMean = sum(listOfConvertedValues) / len(self[column_name])
        xSquared = sum(listOfConvertedValues)*sum(listOfConvertedValues)
        stDev = ((xSquared/len(listOfConvertedValues) - realMean*realMean))**.5
        return stDev




    def __getitem__(self, item):
        # this is for rows only
        if isinstance(item, (int, slice)):
            return self.data[item]

        # this is for columns only
        elif isinstance(item, (str, unicode)):
            return [row[item] for row in self.data]

        # this is for rows and columns
        elif isinstance(item, tuple):
            if isinstance(item[0], list) or isinstance(item[1], list):

                if isinstance(item[0], list):
                    rowz = [row for index, row in enumerate(self.data) if index in item[0]]
                else:
                    rowz = self.data[item[0]]

                if isinstance(item[1], list):
                    if all([isinstance(thing, int) for thing in item[1]]):
                        return [[column_value for index, column_value in enumerate([value for value in row.itervalues()]) if index in item[1]] for row in rowz]
                    elif all([isinstance(thing, (str, unicode)) for thing in item[1]]):
                        return [[row[column_name] for column_name in item[1]] for row in rowz]
                    else:
                        raise TypeError('What the hell is this?')

                else:
                    return [[value for value in row.itervalues()][item[1]] for row in rowz]
            else:
                if isinstance(item[1], (int, slice)):
                    return [[value for value in row.itervalues()][item[1]] for row in self.data[item[0]]]
                elif isinstance(item[1], (str, unicode)):
                    return [row[item[1]] for row in self.data[item[0]]]
                else:
                    raise TypeError('I don\'t know how to handle this...')

        # only for lists of column names
        elif isinstance(item, list):
            return [[row[column_name] for column_name in item] for row in self.data]

    def get_rows_where_column_has_value(self, column_name, value, index_only=False):
        if index_only:
            return [index for index, row_value in enumerate(self[column_name]) if row_value==value]
        else:
            return [row for row in self.data if row[column_name]==value]


def convertToFloatOrDatetime(string):
    try:
        return float(string)
    except:
        return parse(string)

infile = open('SalesJan2009.csv')
lines = infile.readlines()
lines = lines[0].split('\r')
data = [l.split(',') for l in lines]
things = lines[559].split('"')
data[559] = things[0].split(',')[:-1] + [things[1]] + things[-1].split(',')[1:]


df = DataFrame(list_of_lists=data)
# get the 5th row
fifth = df[4]
sliced = df[4:10]

# get item definition for df [row, column]

# get the third column
tupled = df[:, 2]
tupled_slices = df[0:5, :3]

tupled_bits = df[[1, 4], [1, 4]]


# adding header for data with no header
df = DataFrame(list_of_lists=data[1:], header=False)

# fetch columns by name
named = df['column1']
named_multi = df[['column1', 'column7']]

#fetch rows and (columns by name)
named_rows_and_columns = df[:5, 'column7']
named_rows_and_multi_columns = df[:5, ['column4', 'column7']]


#testing from_csv class method
df = DataFrame.from_csv('SalesJan2009.csv')
df['Price']
# testing dups in header exception
# dfDup = DataFrame.from_csv('SalesJan2009withdupheader.csv')
rows = df.get_rows_where_column_has_value('Payment_Type', 'Visa')
indices = df.get_rows_where_column_has_value('Payment_Type', 'Visa', index_only=True)

rows_way2 = df[indices, ['Product', 'Country']]

2+2