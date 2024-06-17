import unittest
import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
ENDPOINT = 'http://176.123.163.78:5000'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


class TestApi(unittest.TestCase):
    def test_home(self):
        resp = requests.get(ENDPOINT)
        self.assertIn('Housing price service', resp.text)

    def test_api(self):
        data = {'author': 'Verges of estate', 'author_type': 'real_estate_agent', 'url': 'https://www.cian.ru/sale/flat/294966171/', 'location': 'Москва', 'deal_type': 'sale', 'accommodation_type': 'flat', 'floor': 4, 'floors_count': 30, 'rooms_count': 3, 'total_meters': 115.6, 'district': 'Южнопортовый', 'street': ' 1-я Машиностроения', 'house_number': '10', 'underground': 'Дубровка', 'residential_complex': 'Дубровская Слобода', 'url_id': 294966171, 'first_floor': False, 'last_floor': False, 'area': 115.6}
        resp = requests.post(ENDPOINT +'/predict',
                             json=data,
                             headers=HEADERS)
        self.assertIn('price', resp.text)
        print(resp.text)


if __name__ == '__main__':
    unittest.main()
