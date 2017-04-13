from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
from dateutil.parser import parse
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import scipy as sp

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.cluster import DBSCAN

# Lets open up the web page
driver = webdriver.Chrome()
driver.get('https://www.google.com/flights/explore/')


# TASK 1
def scrape_data(start_date,from_place,to_place,city_name):

    # Start Date
    array_of_url = driver.current_url.split('d=')
    array_of_url.__delitem__(1)
    array_of_url.append('d=' + start_date)

    new_url = ''.join(array_of_url)
    driver.get(new_url)

    time.sleep(2.5)

    # From Location
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]')
    from_input.click()
    from_input.send_keys(from_place)

    # To Location
    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()
    to_input.send_keys(to_place)

    actions = ActionChains(driver)
    actions.send_keys(from_place)
    actions.send_keys(to_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()

    # Get Data
    city_finder = driver.find_element_by_class_name('LJTSM3-v-c')
    if city_name in city_finder:
        results = driver.find_elements_by_class_name('LJTSM3-v-d')
        test = results[0]
        bars = test.find_elements_by_class_name('LJTSM3-w-x')

    data = []
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.001)
        data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
                     test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))

    # Clean Data
    d = data[0]
    clean_data = [(float(d[0].replace('$', '').replace(',', '')),
                   (parse(d[1].split('-')[0].strip()) - datetime.datetime(2017, 3, 13, 0, 0)).days) for d in data]

    # Return df object w/ 2 columns
    df = pd.DataFrame(clean_data, columns=['Price', 'Start_Date'])
    return df


#TASK 2
def scape_date_90(start_date,from_place,to_place,city_name):

    # Start Date
    array_of_url = driver.current_url.split('d=')
    array_of_url.__delitem__(1)
    array_of_url.append('d=' + start_date)

    new_url = ''.join(array_of_url)
    driver.get(new_url)

    time.sleep(2.5)

    # From Location
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]')
    from_input.click()
    from_input.send_keys(from_place)

    # To Location
    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()
    to_input.send_keys(to_place)

    actions = ActionChains(driver)
    actions.send_keys(from_place)
    actions.send_keys(to_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()

    # Get Data
    city_finder = driver.find_element_by_xpath('LJTSM3-v-c')
    if city_name in city_finder:
        results = driver.find_elements_by_class_name('LJTSM3-v-d')
        test = results[0]
        bars = test.find_elements_by_class_name('LJTSM3-w-x')

    data = []
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.001)
        data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
                     test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))

    d = data[0]
    clean_data = [(float(d[0].replace('$', '').replace(',', '')),
                    (parse(d[1].split('-')[0].strip()) - datetime.datetime(2017, 3, 13, 0, 0)).days) for d in data]

    # df object w/ 2 columns
    df = pd.DataFrame(clean_data, columns=['Price', 'Start_Date'])

    # After 60 bars scanned
    find_more_days = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[4]/div/div[2]/div[1]/div/div[2]/div[2]/div/div[2]/div[5]')
    find_more_days.click()

    # Need to iterate through an extra 30 days here
    data = []
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.001)
        data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
                     test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
    d = data[0]
    clean_data = [(float(d[0].replace('$', '').replace(',', '')),
                   (parse(d[1].split('-')[0].strip()) - datetime.datetime(2017, 3, 13, 0, 0)).days) for d in data]

    # Clean-up extra 30 days and append to original 60 day df
    df2 = pd.DataFrame(clean_data[:30], columns=['Price', 'Start_Date'])
    return df.append(df2)

# TASK 3 Find Mistake Price Here

def task_3_dbscan(flight_data):

    # Code directly from the book chapter
    # Used Epsilon in your code, tried to manipulate the epsilon later on, to get back to my original issue
    X = np.concatenate([days[:, None], prices[:, None]], axis=1)
    db = DBSCAN(eps=30, min_samples=5).fit(X)

    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12, 8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14,
              y=1.01)

    # Return 5 Day period with lowest average price for clusters of more than 5
    # Noise points given -1 DB value
    ## if db.labels == -1:
    ##    return

    days = np.arange(60)
    prices1 = np.random.normal(0, 35, size=20) + 400
    prices2 = np.random.normal(0, 35, size=20) + 800
    prices3 = np.random.normal(0, 35, size=20) + 400
    prices = np.concatenate([prices1, prices2, prices3], axis=0)

    # Introducing a Test Point
    prices[30] = 652
    plt.scatter(days, prices)
    plt.plot(30, 652, 'or')

    # Distance from each cluster, as displayed in Jupyter notebook
    lbls = np.unique(db.labels_)
    cluster_means = [np.mean(X[labels == num, :], axis=0) for num in range(lbls[-1] + 1)]
    noise_point = X[30, :]

    # euclidean
    dist = [euclidean(noise_point, cm) for cm in cluster_means]

    # chebyshev
    dist = [chebyshev(noise_point, cm) for cm in cluster_means]

    # cityblock
    dist = [cityblock(noise_point, cm) for cm in cluster_means]


# let's create some helper functions
    def calculate_cluster_means(X, labels):
        lbls = np.unique(labels)
        print "Cluster labels: {}".format(np.unique(lbls))
        cluster_means = [np.mean(X[labels == num, :], axis=0) for num in range(lbls[-1] + 1)]
        print "Cluster Means: {}".format(cluster_means)
        return cluster_means

    def print_3_distances(noise_point, cluster_means):
        # euclidean
        dist = [euclidean(noise_point, cm) for cm in cluster_means]
        print "Euclidean distance: {}".format(dist)
        # chebyshev
        dist = [chebyshev(noise_point, cm) for cm in cluster_means]
        print "Chebysev distance: {}".format(dist)
        # cityblock
        dist = [cityblock(noise_point, cm) for cm in cluster_means]
        print "Cityblock (Manhattan) distance: {}".format(dist)

    def plot_the_clusters(X, dbscan_model, noise_point=None):
        labels = dbscan_model.labels_
        clusters = len(set(labels))
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12, 8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    if noise_point is not None:
        plt.plot(noise_point[0], noise_point[1], 'xr')

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)

    def do_yo_thang(X, dbscan_model, noise_point):
        cluster_means = calculate_cluster_means(X, dbscan_model.labels_)
        print_3_distances(noise_point, cluster_means)
        plot_the_clusters(X, dbscan_model, noise_point)

    # Fit Clusters plot, after testing various epsilon values
    X_ss = StandardScaler().fit_transform(X)
    db_ss = DBSCAN(eps=0.4, min_samples=3).fit(X_ss)
    noise_point = X_ss[30, :]
    do_yo_thang(X_ss, db_ss, noise_point)

    # Using Stack Overflow Link, found a way to take a snapshot of clusters plot as shown in book
    plt.savefig('task_3_dbscan.png', bbox_inches='tight')

def task_3_IQR(flight_data):

    #IQR Method using Mean Distances

    # Plot Box Plot for Price
    df['Price'].plot.box()
    # Save Box Plot Price
    plt.savefig('task_3_iqr.png.', bbox_inches='tight')

def task_3_ex(flight_data):


# Task 4, what I used orginally to test epsilon for Task 3 the first time I pushed my code
# Cheapest Period to Fly
def task_4_dbscan(flight_data):
    # Tried to pick a fairly stable epsilon, looking at minimal noise
    X = StandardScaler().fit_transform(df[['Start_Date', 'Price']])
    db = DBSCAN(eps=.15, min_samples=3).fit(X)

    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12, 8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)
    df['dbscan_labels'] = db.labels_

    df.head()
    df.dbscan_labels.unique()
    t = X[df.dbscan_labels == 1, :]
    t.mean(axis=0)
    return df