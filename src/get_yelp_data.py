from numpy import genfromtxt
import numpy as np
import argparse
import pdb
import csv

def get_review_data(csv_file):
    names = ('funny', 'user_id', 'review_id', 'text', 'business_id', \
                 'stars', 'date', 'useful', 'cool')
    filling = None

    print "getting review data"
    data = genfromtxt(csv_file, dtype=None, names=names, \
                             delimiter='\t', skip_header=1, comments=None, \
                             filling_values=filling)
    print "done getting review data"
    return data

def get_business_data(csv_file):
    names = ('attributes.Ambience.divey', 'attributes.RestaurantsDelivery', \
        'attributes.DogsAllowed', 'postal_code', 'hours.Thursday', \
        'attributes.HairSpecializesIn.coloring', 'attributes.BestNights.sunday', \
        'attributes.BYOB', 'attributes.AgesAllowed', 'attributes.Music.video', \
        'hours.Friday', 'latitude', 'attributes.Alcohol', \
        'attributes.Ambience.classy', 'attributes.RestaurantsTableService', \
        'business_id', 'attributes.Ambience.touristy', \
        'attributes.RestaurantsCounterService', 'attributes.Corkage', \
        'attributes.RestaurantsGoodForGroups', 'categories', 'name', \
        'attributes.BusinessAcceptsBitcoin', 'attributes.HappyHour', \
        'attributes.WheelchairAccessible', 'attributes.Ambience.hipster', \
        'attributes.BusinessAcceptsCreditCards', 'is_open', \
        'attributes.DietaryRestrictions.vegetarian', 'attributes.Music.live', \
        'attributes.Music.background_music', 'neighborhood', \
        'attributes.BusinessParking.lot', 'attributes.Music.karaoke', \
        'review_count', 'attributes.GoodForMeal.breakfast', \
        'attributes.NoiseLevel', 'attributes.HairSpecializesIn.perms', \
        'state', 'attributes.DriveThru', 'attributes.HasTV', \
        'attributes.GoodForMeal.dinner', 'attributes.BusinessParking.street', \
        'address', 'attributes.RestaurantsAttire', 'hours.Sunday', \
        'attributes.BestNights.tuesday', 'attributes.AcceptsInsurance', \
        'attributes.BestNights.wednesday', 'hours.Wednesday', \
        'attributes.HairSpecializesIn.kids', 'attributes.Open24Hours', \
        'attributes.Ambience.trendy', 'attributes.CoatCheck', 'hours.Monday', \
        'attributes.HairSpecializesIn.straightperms', 'city', \
        'attributes.HairSpecializesIn.curly', 'attributes.Music.no_music', \
        'hours.Tuesday', 'attributes.HairSpecializesIn.africanamerican', \
        'stars', 'attributes.RestaurantsPriceRange2', \
        'attributes.Ambience.intimate', 'attributes.GoodForMeal.latenight', \
        'attributes.GoodForMeal.dessert', 'attributes.BusinessParking.validated', \
        'attributes.GoodForMeal.lunch', 'attributes.GoodForKids', \
        'attributes.DietaryRestrictions.soy-free', 'attributes.GoodForMeal.brunch', \
        'attributes.BusinessParking.valet', 'longitude', \
        'attributes.DietaryRestrictions.gluten-free', \
        'attributes.BYOBCorkage', 'attributes.BusinessParking.garage', \
        'attributes.BestNights.friday', 'hours.Saturday', 'attributes.Music.dj', \
        'attributes.HairSpecializesIn.extensions', 'attributes.BestNights.saturday', \
        'attributes.Ambience.casual', 'attributes.BestNights.thursday', \
        'attributes.BestNights.monday', 'attributes.HairSpecializesIn.asian', \
        'attributes.DietaryRestrictions.kosher', 'attributes.WiFi', 'attributes.Smoking', \
        'attributes.DietaryRestrictions.halal', 'attributes.GoodForDancing', \
        'attributes.ByAppointmentOnly', 'attributes.Caters', \
        'attributes.RestaurantsReservations', 'attributes.DietaryRestrictions.dairy-free', \
        'attributes.DietaryRestrictions.vegan', 'attributes.Ambience.romantic', \
        'attributes.Music.jukebox', 'attributes.Ambience.upscale', \
        'attributes.RestaurantsTakeOut', 'attributes.BikeParking', \
        'attributes.OutdoorSeating');
    filling = None

    print "getting business data"
    data = genfromtxt(csv_file, dtype=None, names=names, \
                             delimiter='\t', skip_header=1, comments=None, \
                             filling_values=filling)
    print "done getting business data"
    return data

def construct_filtered_set(txt_file_path):
    title = ""
    filtered_values = set()
    first_line = True
    with open(txt_file_path, 'r') as fin:
        #while line = fin.readline():
        for line in fin:
            if first_line:
                title = line.strip()
                first_line = False
            else:
                filtered_values.add(line.strip())
    return title, filtered_values

def get_filtered_review_data(review_data, filter_column_index, filtered_value_set):
    filtered_reviews = []
    for review in review_data:
        if review[filter_column_index] in filtered_value_set:
            filtered_reviews.append(review)
    return filtered_reviews

def write_reviews_to_csv_file(input_data, csv_file_path, delimiter='\t'):
    names = ('funny', 'user_id', 'review_id', 'text', 'business_id', \
                 'stars', 'date', 'useful', 'cool')
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout, delimiter=delimiter)
        csv_file.writerow(names)
        for review in input_data:
            csv_file.writerow(review)    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Get Yelp data.',
            )
    parser.add_argument(
            'file_type',
            type=str,
            help='Review or business',
            )
    parser.add_argument(
            'csv_file',
            type=str,
            help='The csv file to load.',
            )
    parser.add_argument(
            'filter_values',
            type=str,
            help='The text files of values to filter reviews.',
            )
    parser.add_argument(
            '--filtered_file_name',
            type=str,
            help='Filepath for csv file containing filtered reviews.',
            )

    args = parser.parse_args()
    file_type = args.file_type
    csv_file = args.csv_file
    filter_values = args.filter_values

    if file_type == 'review':
        data = get_review_data(csv_file)

        if args.filtered_file_name:
            filtered_file_name = args.filtered_file_name
            print "constructing filtered set"
            label, business_ids = construct_filtered_set(filter_values)
            if label == "business_id":
                col_index = 4
                print "getting filtered review data"
                filtered_data = get_filtered_review_data(data, col_index, business_ids)
                print "writing filtered review data to csv file"
                write_reviews_to_csv_file(filtered_data, filtered_file_name)

    elif file_type == 'business':
        data = get_business_data(csv_file)

