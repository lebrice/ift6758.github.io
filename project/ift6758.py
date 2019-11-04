#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import textwrap

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        """)

    parser.add_argument(
        "-i", "--idir",
        required=True, nargs="+",
        help="")

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="")

    return parser.parse_args()

def get_sub_ids(input_dir):
    pic_list = glob.glob(os.path.join(input_dir, 'Image', '*.jpg'))
    test_userids = [os.path.basename(pic).split(".")[0] for pic in pic_list]
    return test_userids

class average_user :

    def __init__(self):
        self.userid = 'placeholder'
        self.age_group = 'xx-24'
        self.gender = getGender
        self.ope = 3.91
        self.con = 3.45
        self.ext = 3.49
        self.agr = 3.58
        self.neu = 2.73

    def to_xml(self):
        user_text = textwrap.dedent(f"""\
        <user
            id="{self.userid}"
            age_group="{self.age_group}"
            gender="{self.gender}"
            extrovert="{self.ext}"
            neurotic="{self.neu}"
            agreeable="{self.agr}"
            conscientious="{self.con}"
            open="{self.ope}"
        />""")
        return user_text
    
    def getGender:
        #TODO PATH_IMAGE + PATH_PROFILES
        profiles = pd.read_csv(PATH_PROFILES)
        oxford = pd.read_csv(PATH_IMAGES)
        
        oxford = oxford.rename(columns={"userId":"userid"})
        oxford.drop_duplicates(subset ="userid",keep = "first", inplace=True)
        
        facial_hair=oxford.loc[:,['userid']]
        facial_hair['hair']= oxford.facialHair_sideburns+oxford.facialHair_mustache+oxford.facialHair_beard
        
        t_gender = profiles.merge(facial_hair, on="userid")
        t_gender.set_index(keys='userid', inplace=True, drop=True)
        t_gender.drop(['Unnamed: 0'], axis=1).head()
        
        Male=t_gender[['hair']][t_gender.loc[:,'gender'] == 0]
        Female=t_gender[['hair']][t_gender.loc[:,'gender'] == 1]


def main():
    args = get_arguments()
    oFolder = args.odir[0]
    iFolder = args.idir[0]

    # Create oFolder if not exists
    if not os.path.exists(oFolder):
        os.mkdir(oFolder)

    id_list = get_sub_ids(iFolder)
    #print(id_list)

    user =  average_user()

    for id in id_list:
        user.userid = id
        #print(user.to_xml())
        with open(os.path.join(oFolder, f"{id}.xml"), "w") as xml_file:
            xml_file.write(user.to_xml())

if __name__ == '__main__':
    sys.exit(main())
