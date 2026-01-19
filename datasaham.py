"""
Datasaham.io API Client
Indonesian Stock Market Data API
"""

import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class DatasahamAPI:
    """Client untuk mengakses Datasaham.io API - Data Saham Indonesia"""

    BASE_URL = "https://api.datasaham.io/api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Content-Type": "application/json"
        })

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Base request method"""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise Exception(data.get("error", "Unknown error"))

        return data.get("data", data)

    # ==================== MAIN ====================

    def search(self, query: str) -> Dict[str, Any]:
        """
        Cari saham berdasarkan kode atau nama

        Args:
            query: Kode saham atau nama perusahaan (e.g., "BBCA", "Bank")
        """
        return self._request("main/search", {"q": query})

    def trending(self) -> List[Dict[str, Any]]:
        """Dapatkan daftar saham trending"""
        return self._request("main/trending")

    # ==================== CHART ====================

    def chart_daily(self, symbol: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """
        Data OHLCV harian

        Args:
            symbol: Kode saham (e.g., "BBCA")
            from_date: Tanggal mulai format YYYY-MM-DD (tanggal lebih baru)
            to_date: Tanggal akhir format YYYY-MM-DD (tanggal lebih lama)
        """
        return self._request(f"chart/{symbol}/daily", {
            "from": from_date,
            "to": to_date
        })

    def chart_weekly(self, symbol: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """Data OHLCV mingguan"""
        return self._request(f"chart/{symbol}/weekly", {
            "from": from_date,
            "to": to_date
        })

    def chart_monthly(self, symbol: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """Data OHLCV bulanan"""
        return self._request(f"chart/{symbol}/monthly", {
            "from": from_date,
            "to": to_date
        })

    def chart_intraday(self, symbol: str, interval: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """
        Data OHLCV intraday

        Args:
            symbol: Kode saham
            interval: Interval waktu (15m, 30m, 1h, 2h, 3h, 4h)
            from_date: Tanggal mulai format YYYY-MM-DD
            to_date: Tanggal akhir format YYYY-MM-DD
        """
        return self._request(f"chart/{symbol}/{interval}", {
            "from": from_date,
            "to": to_date
        })

    # ==================== MOVERS ====================

    def top_gainer(self) -> Dict[str, Any]:
        """Saham dengan kenaikan harga tertinggi"""
        return self._request("movers/top-gainer")

    def top_loser(self) -> Dict[str, Any]:
        """Saham dengan penurunan harga terbesar"""
        return self._request("movers/top-loser")

    def top_value(self) -> Dict[str, Any]:
        """Saham dengan nilai transaksi tertinggi"""
        return self._request("movers/top-value")

    def top_volume(self) -> Dict[str, Any]:
        """Saham dengan volume transaksi tertinggi"""
        return self._request("movers/top-volume")

    def top_frequency(self) -> Dict[str, Any]:
        """Saham dengan frekuensi transaksi tertinggi"""
        return self._request("movers/top-frequency")

    def net_foreign_buy(self) -> Dict[str, Any]:
        """Saham dengan net foreign buy tertinggi"""
        return self._request("movers/net-foreign-buy")

    def net_foreign_sell(self) -> Dict[str, Any]:
        """Saham dengan net foreign sell tertinggi"""
        return self._request("movers/net-foreign-sell")

    # ==================== SECTORS ====================

    def sectors(self) -> Dict[str, Any]:
        """Daftar semua sektor"""
        return self._request("sectors")

    def sector_stocks(self, sector_id: str) -> Dict[str, Any]:
        """Daftar saham dalam sektor tertentu"""
        return self._request(f"sectors/{sector_id}/stocks")

    def sector_performance(self, sector_id: str) -> Dict[str, Any]:
        """Performa sektor"""
        return self._request(f"sectors/{sector_id}/performance")

    # ==================== CALENDAR ====================

    def calendar_dividend(self) -> Dict[str, Any]:
        """Jadwal pembagian dividen"""
        return self._request("calendar/dividend")

    def calendar_stocksplit(self) -> Dict[str, Any]:
        """Jadwal stock split"""
        return self._request("calendar/stocksplit")

    def calendar_ipo(self) -> Dict[str, Any]:
        """Jadwal IPO"""
        return self._request("calendar/ipo")

    def calendar_rups(self) -> Dict[str, Any]:
        """Jadwal RUPS"""
        return self._request("calendar/rups")

    def calendar_rights(self) -> Dict[str, Any]:
        """Jadwal rights issue"""
        return self._request("calendar/rights")

    # ==================== EMITEN ====================

    def emiten_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Profil lengkap perusahaan

        Args:
            symbol: Kode saham (e.g., "BBCA")
        """
        return self._request(f"emiten/{symbol}/profile")

    def emiten_financials(self, symbol: str) -> Dict[str, Any]:
        """Data keuangan perusahaan"""
        return self._request(f"emiten/{symbol}/financials")

    def emiten_broker_summary(self, symbol: str) -> Dict[str, Any]:
        """Ringkasan aktivitas broker"""
        return self._request(f"emiten/{symbol}/broker-summary")

    def emiten_insider(self, symbol: str) -> Dict[str, Any]:
        """Data insider trading"""
        return self._request(f"emiten/{symbol}/insider")

    # ==================== MARKET DETECTOR ====================

    def broker_activity(self, broker_code: str) -> Dict[str, Any]:
        """Aktivitas broker tertentu"""
        return self._request(f"market-detector/broker/{broker_code}/activity")

    def top_broker(self) -> Dict[str, Any]:
        """Top broker berdasarkan aktivitas"""
        return self._request("market-detector/top-broker")

    # ==================== GLOBAL MARKET ====================

    def global_indices(self) -> Dict[str, Any]:
        """Data indeks global"""
        return self._request("global/indices")

    def global_commodities(self) -> Dict[str, Any]:
        """Data komoditas global"""
        return self._request("global/commodities")

    def global_forex(self) -> Dict[str, Any]:
        """Data forex"""
        return self._request("global/forex")

    # ==================== HELPER METHODS ====================

    def get_stock_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Helper: Ambil data saham lengkap untuk N hari terakhir

        Args:
            symbol: Kode saham
            days: Jumlah hari ke belakang (default 30)
        """
        today = datetime.now()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "chart": self.chart_daily(symbol, from_date, to_date),
            "profile": self.emiten_profile(symbol)
        }

    def get_market_overview(self) -> Dict[str, Any]:
        """
        Helper: Ringkasan pasar hari ini
        """
        return {
            "trending": self.trending(),
            "top_gainer": self.top_gainer(),
            "top_loser": self.top_loser(),
            "top_value": self.top_value()
        }


# Contoh penggunaan
if __name__ == "__main__":
    API_KEY = "sbk_a05e1010d49474dba301908a72499aa96c83ee8ef869c4b3"

    api = DatasahamAPI(API_KEY)

    print("=" * 60)
    print("DATASAHAM.IO API CLIENT - Demo")
    print("=" * 60)

    # 1. Search saham
    print("\n[1] Mencari saham BBCA...")
    result = api.search("BBCA")
    companies = result.get("data", {}).get("company", [])[:3]
    for c in companies:
        print(f"    - {c['name']}: {c['desc']}")

    # 2. Trending stocks
    print("\n[2] Saham trending hari ini...")
    trending = api.trending()
    trending_list = trending if isinstance(trending, list) else trending.get("data", trending)
    if isinstance(trending_list, dict):
        trending_list = trending_list.get("data", [])
    for stock in trending_list[:5]:
        print(f"    - {stock['symbol']}: {stock['name']} ({stock['change']} | {stock['percent']}%)")

    # 3. Top gainers
    print("\n[3] Top gainers...")
    gainers = api.top_gainer()
    for stock in gainers.get("data", {}).get("mover_list", [])[:5]:
        detail = stock["stock_detail"]
        change = stock["change"]
        print(f"    - {detail['code']}: {detail['name']} (+{change['percentage']:.2f}%)")

    # 4. Top losers
    print("\n[4] Top losers...")
    losers = api.top_loser()
    for stock in losers.get("data", {}).get("mover_list", [])[:5]:
        detail = stock["stock_detail"]
        change = stock["change"]
        print(f"    - {detail['code']}: {detail['name']} ({change['percentage']:.2f}%)")

    # 5. Chart data
    print("\n[5] Data harian BBCA (5 hari terakhir)...")
    chart = api.chart_daily("BBCA", "2026-01-19", "2026-01-01")
    for candle in chart.get("data", {}).get("chartbit", [])[:5]:
        print(f"    - {candle['date']}: O={candle['open']} H={candle['high']} L={candle['low']} C={candle['close']} V={candle['volume']}")

    # 6. Dividend calendar
    print("\n[6] Jadwal dividen mendatang...")
    dividends = api.calendar_dividend()
    for div in dividends.get("data", {}).get("dividend", [])[:5]:
        print(f"    - {div['company_symbol']}: {div['dividend_value_formatted']} (Ex-date: {div['dividend_exdate']})")

    # 7. Sectors
    print("\n[7] Daftar sektor...")
    sectors = api.sectors()
    for sector in sectors.get("data", [])[:10]:
        print(f"    - {sector['id']}: {sector['name']}")

    # 8. Company profile
    print("\n[8] Profil BBCA...")
    profile = api.emiten_profile("BBCA")
    print(f"    Background: {profile.get('background', '')[:200]}...")

    print("\n" + "=" * 60)
    print("Demo selesai!")
    print("=" * 60)
