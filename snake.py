import os
import MetaTrader5 as mt5
import random
from datetime import datetime, time as dt_time
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Importar Random Forest
from statsmodels.tsa.arima.model import ARIMA
import tkinter as tk
from tkinter import messagebox
import logging
from sklearn.model_selection import TimeSeriesSplit

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conexión a MetaTrader 5
if not mt5.initialize():
    logger.error("No se pudo iniciar MetaTrader 5")
    mt5.shutdown()

def mercado_abierto():
    """Verifica si el mercado está abierto."""
    hora_actual = datetime.now().time()
    inicio_hora = dt_time(9, 0)
    fin_hora = dt_time(17, 0)
    dia_actual = datetime.now().weekday()  # 0 = lunes, 6 = domingo
    return dia_actual < 5 and inicio_hora <= hora_actual <= fin_hora

def obtener_datos_historicos(simbolo, timeframe, cantidad):
    """Obtiene datos históricos de precios para un símbolo y un marco de tiempo específicos."""
    rates = mt5.copy_rates_from_pos(simbolo, timeframe, 0, cantidad)
    if rates is None:
        raise ValueError(f"No se pudieron obtener datos para el símbolo: {simbolo}")
    return rates

class ForexAnalyzer:
    def __init__(self):
        self.price_data = {}
        self.analysis_results = {}
        self.model = LinearRegression()

    def load_historical_data(self, file_path):
        """Carga datos históricos desde un archivo CSV."""
        try:
            historical_data = pd.read_csv(file_path)
            historical_data['Date'] = pd.to_datetime(historical_data['Date'])
            historical_data.set_index('Date', inplace=True)
            self.price_data = historical_data
            logger.info("Datos históricos cargados correctamente.")
        except Exception as e:
            logger.error(f"Error al cargar datos históricos: {str(e)}")

    def train_model(self, prices):
        X = np.array(range(len(prices))).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        self.model.fit(X, y)

    def cross_validate(self, prices, n_splits=5):
        """Realiza validación cruzada en los datos de precios."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_index, test_index in tscv.split(prices):
            X_train, X_test = np.array(range(len(prices)))[train_index].reshape(-1, 1), np.array(range(len(prices)))[test_index].reshape(-1, 1)
            y_train, y_test = prices[train_index], prices[test_index]

            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
            logger.info(f"Score de validación cruzada: {score:.4f}")

        return np.mean(scores)

    def predict_price(self, last_index):
        next_index = np.array([[last_index + 1]])
        predicted_price = self.model.predict(next_index)
        return predicted_price[0][0]

    def calculate_ema(self, prices, period):
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(prices, weights, mode='valid')[-1]

    def calculate_bollinger_bands(self, prices, window=20, num_std_dev=2):
        """Calcula las Bandas de Bollinger para una serie de precios."""
        if len(prices) < window:
            logger.warning("No hay suficientes datos para calcular las Bandas de Bollinger.")
            return None, None

        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)

        return upper_band.iloc[-1], lower_band.iloc[-1]

    def identify_trend(self, timeframe):
        if timeframe in ['D1', 'H4', 'H1', 'M15', 'M5']:
            prices = self.price_data[timeframe]['Price'].values  # Cambiado de 'Close' a 'Price'
            if len(prices) < 22:
                logger.warning(f"No hay suficientes datos para calcular la tendencia en {timeframe}.")
                return None
            ema_22 = self.calculate_ema(prices, 22)
            ema_8 = self.calculate_ema(prices, 8)
            trend = 'up' if ema_8 > ema_22 else 'down'
            self.price_data[timeframe]['trend'] = trend
            return trend

    def arima_forecast(self, prices):
        model = ARIMA(prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]

    def perform_multiple_analysis(self):
        self.analysis_results = {
            'analysis_1': (self.identify_trend('D1'), self.identify_trend('H4')),
            'analysis_2': (self.identify_trend('H1'), self.identify_trend('M15')),
            'analysis_3': (self.identify_trend('M5'),)
        }

        upper_band, lower_band = self.calculate_bollinger_bands(self.price_data['M5']['Price'].values)
        self.price_data['M5']['upper_band'] = upper_band
        self.price_data['M5']['lower_band'] = lower_band

        return self.analysis_results

    def determine_entry_point(self, probability_threshold=70):
        for analysis, trends in self.analysis_results.items():
            if all(trend == 'down' for trend in trends if trend is not None):
                probability = random.randint(1, 99)
                if probability >= probability_threshold:
                    if self.price_data['M5']['Price'].values[-1] > self.price_data['M5']['upper_band']:
                        return {
                            'direction': 'sell',
                            'entry_level': self.price_data['M5']['Price'].values[-1] - 0.001,
                            'probability': probability,
                            'order_type': 'limit'
                        }
            elif all(trend == 'up' for trend in trends if trend is not None):
                probability = random.randint(1, 99)
                if probability >= probability_threshold:
                    if self.price_data['M5']['Price'].values[-1] < self.price_data['M5']['lower_band']:
                        return {
                            'direction': 'buy',
                            'entry_level': self.price_data['M5']['Price'].values[-1] + 0.001,
                            'probability': probability,
                            'order_type': 'stop'
                        }
        return None

    def calculate_stop_loss(self, entry_direction, risk_amount):
        if entry_direction == 'sell':
            return self.price_data['M5']['recent_high'] + risk_amount
        elif entry_direction == 'buy':
            return self.price_data['M5']['recent_low'] - risk_amount

    def set_take_profit_levels(self, entry_direction, risk_amount):
        levels = []
        if entry_direction == 'sell':
            levels.append(self.price_data['M5']['key_level'] - (2 * risk_amount))  # 1:2
            levels.append(self.price_data['M5']['key_level'] - (3 * risk_amount))  # 1:3
            levels.append(self.price_data['M5']['key_level'] - (1.2 * risk_amount))  # 1:1.2
        elif entry_direction == 'buy':
            levels.append(self.price_data['M5']['key_level'] + (2 * risk_amount))  # 1:2
            levels.append(self.price_data['M5']['key_level'] + (3 * risk_amount))  # 1:3
            levels.append(self.price_data['M5']['key_level'] + (1.2 * risk_amount))  # 1:1.2
        return levels

    def set_parameters(self, stop_loss, take_profit, ema_period):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.ema_period = ema_period

class RandomForestAnalyzer(ForexAnalyzer):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # Inicializar el modelo Random Forest

    def train_model(self, prices):
        X = np.array(range(len(prices))).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        self.model.fit(X, y)

    def predict_price(self, last_index):
        next_index = np.array([[last_index + 1]])
        predicted_price = self.model.predict(next_index)
        return predicted_price[0][0]

def mostrar_alerta(simbolo, entry, stop_loss, take_profit_levels, predicted_price):
    mensaje = f"""¡Oportunidad de trading detectada!

Divisa: {simbolo}
Punto de Entrada: {entry['entry_level']:.5f}
Tipo de Entrada: {entry['order_type'].capitalize()}
Take Profit: {', '.join([f'{tp:.5f}' for tp in take_profit_levels])}
Stop Loss: {stop_loss:.5f}
Probabilidad de Ganancia: {entry['probability']:.2f}%
Precio Predicho (ARIMA): {predicted_price:.5f}
"""
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    messagebox.showinfo("Alerta de GONA_TRADING", mensaje)
    root.destroy()

def obtener_datos_simulados(simbolo):
    try:
        rates_d1 = obtener_datos_historicos(simbolo, mt5.TIMEFRAME_D1, 100)
        rates_h4 = obtener_datos_historicos(simbolo, mt5.TIMEFRAME_H4, 100)
        rates_h1 = obtener_datos_historicos(simbolo, mt5.TIMEFRAME_H1, 100)
        rates_m15 = obtener_datos_historicos(simbolo, mt5.TIMEFRAME_M15, 100)
        rates_m5 = obtener_datos_historicos(simbolo, mt5.TIMEFRAME_M5, 100)

        price_data = {
            'D1': {'Price': rates_d1['close']},
            'H4': {'Price': rates_h4['close']},
            'H1': {'Price': rates_h1['close']},
            'M15': {'Price': rates_m15['close']},
            'M5': {
                'Price': rates_m5['close'],
                'key_level': rates_m5[-1]['close'],
                'recent_high': rates_m5[-1]['high'],
                'recent_low': rates_m5[-1]['low'],
                'support_level': rates_m5[-1]['low'],
                'resistance_level': rates_m5[-1]['high'],
                'trend': None
            }
        }
        return price_data
    except Exception as e:
        logger.error(f"Error al obtener datos simulados para {simbolo}: {str(e)}")
        return None

def generar_reporte_divisas() -> None:
    simbolos = ["EURUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"]
    report_number = 1

    for simbolo in simbolos:
        try:
            logger.info(f"Analizando {simbolo}...")
            price_data = obtener_datos_simulados(simbolo)
            if price_data is None:
                logger.error(f"No se pudieron obtener datos para {simbolo}.")
                continue

            analyzer = RandomForestAnalyzer()  # Usar el nuevo analizador de Random Forest
            analyzer.load_historical_data(r'C:\Users\wsanchez\Desktop\Portfolio-React-Tailwind-master\public\csv.csv')  # Cargar datos históricos

            # Entrenar el modelo con precios de M5 antes de realizar el análisis
            analyzer.train_model(price_data['M5']['Price'])
            # Realizar validación cruzada
            cross_val_score = analyzer.cross_validate(price_data['M5']['Price'])
            logger.info(f"Score promedio de validación cruzada para {simbolo}: {cross_val_score:.4f}")

            analyzer.perform_multiple_analysis()
            entry = analyzer.determine_entry_point()

            reporte = f"""
            ===================================
                      REPORTE DE ANÁLISIS {report_number}
            ===================================
            Divisa: {simbolo}
            Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Tendencias:
            - D1: {price_data['D1']['trend']}
            - H4: {price_data['H4']['trend']}
            - H1: {price_data['H1']['trend']}
            - M15: {price_data['M15']['trend']}
            - M5: {price_data['M5']['trend']}
            """

            if entry:
                risk_amount = 0.0008  # Define la cantidad de riesgo
                stop_loss = analyzer.calculate_stop_loss(entry['direction'], risk_amount)
                take_profit_levels = analyzer.set_take_profit_levels(entry['direction'], risk_amount)
                predicted_price = analyzer.arima_forecast(price_data['M5']['Price'])  # Usar ARIMA para la predicción

                # Mostrar alerta emergente si la probabilidad es mayor a 75
                if entry['probability'] > 75:
                    mostrar_alerta(simbolo, entry, stop_loss, take_profit_levels, predicted_price)

                # Imprimir en consola si la probabilidad es menor a 60
                if entry['probability'] < 60:
                    print(f"Operación con baja probabilidad detectada:\n"
                          f"Divisa: {simbolo}\n"
                          f"Punto de Entrada: {entry['entry_level']:.5f}\n"
                          f"Tipo de Entrada: {entry['order_type'].capitalize()}\n"
                          f"Take Profit: {', '.join([f'{tp:.5f}' for tp in take_profit_levels])}\n"
                          f"Stop Loss: {stop_loss:.5f}\n"
                          f"Probabilidad de Ganancia: {entry['probability']:.2f}%\n"
                          f"Precio Predicho (ARIMA): {predicted_price:.5f}\n")

                reporte += f"""
            Análisis de Entrada:
            - Dirección: {entry['direction'].capitalize()}
            - Tipo de Orden: {entry['order_type'].capitalize()}
            - Punto de Entrada: {entry['entry_level']:.5f}
            - Stop Loss: {stop_loss:.5f}
            - Take Profit Levels: {[f'{tp:.5f}' for tp in take_profit_levels]}
            - Probabilidad de Ganancia: {entry['probability']:.2f}%
            - Precio Predicho (ARIMA): {predicted_price:.5f}
            """

                # Crear alerta si la probabilidad es mayor a 76
                if entry['probability'] > 76:
                    logger.info(f"ALERTA: Probabilidad de ganar muy alta (75% o más) para {simbolo}. ¡Considera abrir una GRAN posición!")

                if entry['probability'] >= 75:
                    reporte += "\nALERTA: Probabilidad de ganar muy alta (75% o más). ¡Considera abrir una GRAN posición!"
                elif entry['probability'] >= 70:
                    reporte += "\nALERTA: Probabilidad de ganar alta (70 a 74%). ¡Considera abrir una posición!"
                else:
                    reporte += "\nProbabilidad de ganar: Baja (menos de 70%)... ¡Cuidado!"
            else:
                reporte += "\nNo se recomienda entrada para este símbolo debido a falta de consistencia en análisis."

            reporte += "\n===================================\n"
            logger.info(reporte)
            report_number += 1
        except Exception as e:
            logger.error(f"Error al procesar {simbolo}: {str(e)}")

def GONZA_TRADER():
    if not mt5.initialize():
        logger.error("No se pudo iniciar MetaTrader 5")
        mt5.shutdown()
    else:
        try:
            while True:
                if mercado_abierto():
                    start_time = time.time()
                    logger.info(f"Ejecutando análisis en {datetime.now().strftime('%Y-%m-%d %H:%M')}...")
                    generar_reporte_divisas()  # Llama a la función para generar el reporte
                else:
                    logger.info("El mercado está cerrado cabezón. Esperando 300 segundos hasta el próximo análisis.")
                
                # Esperar 300 segundos antes del próximo análisis
                time.sleep(300)
        except KeyboardInterrupt:
            logger.info("Deteniendo el análisis...")
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    GONZA_TRADER()