from google.oauth2 import service_account
import gspread
import json
import pandas as pd
import sqlite3

# Assuming cfg.json is in the same directory as this script.
# If it's elsewhere, provide the full path.
# Use the path to your service account JSON file
credentials_path = "cfg.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds_with_scope = credentials.with_scopes(scope)
client = gspread.authorize(creds_with_scope)

spreadsheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1ZUq89oHWoBD6VAYh2ojMwRyyjLVChx35OXwoOGKR21M/edit?pli=1#gid=613788280')
worksheet = spreadsheet.get_worksheet(0)

news_data = worksheet.get_all_records()

news_df = pd.DataFrame.from_dict(news_data)

print(news_df)

news_df.to_csv('/Users/mattdolan/PycharmProjects/python/poc-horizon-scanning/news_data.csv')

print(news_df.columns)