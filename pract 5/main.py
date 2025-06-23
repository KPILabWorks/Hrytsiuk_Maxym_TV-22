from faker import Faker
import pandas as pd
import random

fake = Faker('en_US')
num_records = 100
data = []

for _ in range(num_records):
    record = {
        'Full Name': fake.name(),
        'Email': fake.email(),
        'Phone': fake.phone_number(),
        'City': fake.city(),
        'Company': fake.company(),
        'Registration Date': fake.date_between(start_date='-2y', end_date='today'),
        'Salary': round(random.uniform(30000, 100000), 2)
    }
    data.append(record)


df = pd.DataFrame(data)
df.to_csv('synthetic_data.csv', index=False)

print(df.head())
