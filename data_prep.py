#!/usr/bin/env python3
import datetime as dt
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from pandas_datareader import data as pdr
import cv2
from tqdm import tqdm
import logging
import shutil
import urllib.request as request
from contextlib import closing

yf.pdr_override()
os.chdir(sys.path[0])


def disable_print(): sys.stdout = open(os.devnull, 'w')


def enable_print(): sys.stdout = sys.__stdout__


def percent_change(a, b):
    return (b - a) / a * 100


def main():

    logging.basicConfig(filename=f"data_prep_{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')}.log", format='%(asctime)s %(message)s', level=logging.INFO)

    w, h = 259, 194
    x, y = 87, 41

    mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', inherit=True)

    s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

    with closing(request.urlopen('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt')) as r:
        with open('data/nasdaqlisted.txt', 'wb') as f:
            shutil.copyfileobj(r, f)

    df = pd.read_csv("data/nasdaqlisted.txt", sep="|")

    df = df[(df["Market Category"] == "Q") &
            (df["Test Issue"] == "N") &
            (df["Financial Status"] == "N") &
            (df["ETF"] == "N")]

    symbols = df["Symbol"].values

    print("Number of symbols total:", len(symbols))

    chunk_size = 50

    chunk_count = len(list(range(0, len(symbols), chunk_size)))

    directory = "data/npy"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for chunk_i in range(0, len(symbols), chunk_size):

        symbols_in_chunk = symbols[chunk_i:chunk_i+chunk_size]

        training_data = []

        npy_file_name = f"{directory}/{chunk_i}.npy"

        logging.info(f'Saving chunk {chunk_i} of {chunk_count}, number of symbols in chunk: {len(symbols_in_chunk)} ...')

        for symbol in tqdm(symbols_in_chunk):
            try:
                disable_print()
                df = pdr.get_data_yahoo(symbol, dt.datetime(2000, 1, 1), dt.datetime(2020, 10, 1))
                # df.to_csv(f"data/csv/{symbol}.csv")
                enable_print()
                if df is None or len(df) == 0:
                    raise FileNotFoundError
            except Exception as e:
                logging.info(f"{symbol} failed download")
                continue
            for year in range(1998, 2021):
                for month in range(1, 13, 3):
                    try:
                        df_3_month = df[(df.index.year == year) & (df.index.month >= month) & (df.index.month <= month+2)]
                        if len(df_3_month) < 59:
                            continue

                        df_X = df_3_month[:-6]
                        df_y = df_3_month[-6:]

                        file_name = f"{symbol}_{year}_{month}.png"

                        mpf.plot(df_X, type='candle', style=s, mav=(10, 30), volume=True, axisoff=True, figscale=0.5,
                                 savefig=file_name)

                        img = cv2.imread(file_name)
                        os.remove(file_name)

                        cropped_img = img[y:y+h, x:x+w]

                        perc_change = percent_change(df_y.iloc[0]["Adj Close"], df_y.iloc[-1]["Adj Close"])

                        training_data.append([cropped_img, perc_change])

                        # cv2.imshow("", cropped_img)
                        # cv2.waitKey(0)
                    except Exception as e:
                        pass
                        # logging.info(f"{symbol} failed for year {month} month {month}")

        np.save(npy_file_name, training_data, allow_pickle=True)
        logging.info(f'Saved chunk {chunk_i} of {chunk_count}, length {len(training_data)}')

    logging.info("Success!")


if __name__ == "__main__":
    main()