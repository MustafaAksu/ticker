# sector_map.py
import json
import pathlib

sectors = {
    'THYAO': 'Havayolu',
    'GARAN': 'Banka',
    'AKBNK': 'Banka',
    'ISCTR': 'Banka',
    'YKBNK': 'Banka',
    'ASELS': 'Savunma',
    'EREGL': 'Çelik',
    'SISE': 'Cam',
    'PETKM': 'Kimya',
    'TUPRS': 'Rafineri',
    'SAHOL': 'Holding',
    'KCHOL': 'Holding',
    'BIMAS': 'Perakende',
    'TTKOM': 'Telekom',
    'ARCLK': 'Beyaz Eşya',
    'FROTO': 'Otomotiv',
    'VESTL': 'Beyaz Eşya',
    'TCELL': 'Telekom',
    'HALKB': 'Banka',
    'ENKAI': 'İnşaat'
}

path = pathlib.Path('data/metadata/sector_map.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(sectors, ensure_ascii=False, indent=2),
    encoding='utf-8'  # BU SATIR KRİTİK!
)
print("sector_map.json başarıyla oluşturuldu!")