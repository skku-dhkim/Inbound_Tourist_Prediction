from xlrd import open_workbook
import pandas as pd
import glob
import re
from config.codebook import national_kor_to_eng
from config.db_config import mysql_info
from sqlalchemy import create_engine
import pymysql

# DB connection
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       .format(user=mysql_info['user'], pw=mysql_info['pw'], host=mysql_info['host'], db=mysql_info['db']))

# Load file name
dir_list = glob.glob("./data/*")

# Daily
for dl in dir_list:
    try:
        # Open excel file
        xl = open_workbook(dl)
        # Sheet setting
        x1_sheet = xl.sheet_by_index(0)
        # Set title
        row = x1_sheet.row(0)
        title = ["Date", "Total_dept", "Total_entry"]
        title += [national_kor_to_eng[row[i].value] for i in range(9, len(row), 4)]

        # Date string
        ddate = x1_sheet.name.split("_")[-1]

        dataset = []
        dataset.append(title)

        # Get the data from excel
        for row_idx in range(3, x1_sheet.nrows):
            temp = []
            cell_obj = x1_sheet.cell(row_idx, 0)
            # Regular expression to extract day only
            day = re.findall(r'[0-9]+', cell_obj.value)
            day = "{:02d}".format(int(day[0]))
            date = "{}{}".format(ddate, day)
            temp.append(date)
            for col_idx in range(2, x1_sheet.ncols, 4):
                cell_obj = x1_sheet.cell(row_idx, col_idx)
                temp.append(cell_obj.value)
            dataset.append(temp)
        # To Data frame
        df = pd.DataFrame(dataset[1:], columns=dataset[0])
        # Write into DB
        df.to_sql('T_DAILY_FOREINERS', con=engine, index=False, if_exists='append')
    except pymysql.Error as sql_e:
        print(sql_e)
    except Exception as e:
        print(e)
        print(dl)


# Monthly
for dl in dir_list:
    try:
        # Open excel file
        xl = open_workbook(dl)
        # Sheet setting
        x1_sheet = xl.sheet_by_index(0)
        # Set title
        row = x1_sheet.row(0)
        title = ["Date", "Total_dept", "Total_entry"]
        title += [national_kor_to_eng[row[i].value] for i in range(9, len(row), 4)]

        # Date string
        ddate = x1_sheet.name.split("_")[-1]
        date = '{}00'.format(ddate)

        dataset = []
        dataset.append(title)

        temp = []
        temp.append(date)
        # Get the data from excel
        for col_idx in range(2, x1_sheet.ncols, 4):
            cell_obj = x1_sheet.cell(2, col_idx)
            temp.append(cell_obj.value)
        dataset.append(temp)
        # To Data frame
        df = pd.DataFrame(dataset[1:], columns=dataset[0])
        # Write into DB
        df.to_sql('T_MONTHLY_FOREINERS', con=engine, index=False, if_exists='append')
    except pymysql.Error as sql_e:
        print(sql_e)
    except Exception as e:
        print(e)
        print(dl)

