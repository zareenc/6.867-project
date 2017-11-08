from numpy import genfromtxt
import numpy as np
import argparse
import pdb

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Get Yelp data.',
            )
    parser.add_argument(
            'type',
            type=str,
            help='The csv file to load.',
            )

    args = parser.parse_args()
    file_type = args.type

    csv_file = '../data/%s.csv' % file_type

    if file_type == 'review':
        data = get_review_data(csv_file)
    elif file_type == 'business':
        data = get_business_data(csv_file)
