# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:30:08 2020

@author: sablo
"""

import pandas as pd


def getCourseReviewsbyTag():
    courses = pd.read_csv("Coursera_courses.csv")
    coursedetails = pd.read_csv("coursera-course-detail-data.csv")
    coursedetails['course_id'] = coursedetails['Url'].apply(lambda z: z.split('/')[-1]) 
    courses = courses.merge(coursedetails, how = "left", on='course_id')
    courses.drop(['Unnamed: 0','Name','Url'], inplace=True, axis = 1)
    
    
    coursereviews = pd.read_csv("Coursera_reviews.csv")   
    return coursereviews.merge(courses, how='left', on='course_id')[['Tags','course_id','reviews']]

def getTagsList():
    courses = pd.read_csv("Coursera_courses.csv")
    coursedetails = pd.read_csv("coursera-course-detail-data.csv")
    coursedetails['course_id'] = coursedetails['Url'].apply(lambda z: z.split('/')[-1]) 
    courses = courses.merge(coursedetails, how = "left", on='course_id')
    x = courses.groupby("Tags").count()
    return x['Name']