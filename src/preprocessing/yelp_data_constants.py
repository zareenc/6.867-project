# Column indices
REVIEW_BUSID_IDX = 4
REVIEW_USERID_IDX = 1
BUSINESS_BUSID_IDX = 15
USER_USERID_IDX = 2

# Column names
REVIEW_NAMES = ('funny', 'user_id', 'review_id', 'text', 'business_id', \
        'stars', 'date', 'useful', 'cool')
USER_NAMES = ('funny', 'compliment_more', 'friends', 'useful', 'yelping_since', \
                  'compliment_funny', 'user_id', 'complimen_note', 'compliment_photos', \
                  'average_stars', 'compliment_hot', 'elite', 'fans', 'compliment_plain', \
                  'review_count', 'compliment_writer', 'name', 'compliment_cool', 'cool', \
                  'compliment_cute', 'compliment_list', 'compliment_profile')
'''
USER_NAMES = ('compliment_profile', 'compliment_funny', 'user_id', 'compliment_cute', \
        'friends', 'compliment_writer', 'compliment_list', 'useful', 'compliment_plain', \
        'compliment_note', 'yelping_since', 'cool', 'fans', 'review_count', 'funny', \
        'average_stars', 'compliment_more', 'elite', 'compliment_hot', 'name', \
        'compliment_cool', 'compliment_photos')
'''
BUSINESS_NAMES = ('attributes.Ambience.divey', 'attributes.RestaurantsDelivery', \
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
        'attributes.OutdoorSeating')

# Dictionary
type_to_names = {'review': REVIEW_NAMES,
                 'user': USER_NAMES,
                 'business': BUSINESS_NAMES}
