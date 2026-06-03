from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
import os
import json
import msgpack
import numpy as np
import zlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64


# ============================================================================
# КОНСТАНТЫ И НАСТРОЙКИ
# ============================================================================

class SACConfig:
    """Конфигурация для обработки SAC файлов"""
    DEFAULT_CLIENT = "RESIF"  # FDSN клиент по умолчанию
    DEFAULT_OUTPUT_DIR = "SAC_PROCESSED"
    DEFAULT_CHANNELS = "BH?"  # Широкополосные сейсмические каналы

    # Значения по умолчанию для отсутствующих координат
    UNDEFINED_VALUE = -12345.0


# ============================================================================
# КЛАСС ДЛЯ ЗАГРУЗКИ И ОБРАБОТКИ ДАННЫХ
# ============================================================================

class SACDataProcessor:
    """Основной класс для работы с сейсмическими данными и SAC файлами"""

    def __init__(self, client_name: str = None, output_dir: str = None):
        """
        Инициализация процессора

        Args:
            client_name: имя FDSN клиента (RESIF, IRIS, etc.)
            output_dir: папка для сохранения результатов
        """
        self.client = Client(client_name or SACConfig.DEFAULT_CLIENT)
        self.output_dir = Path(output_dir or SACConfig.DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    # ============================================================================
    # ЭТАП 1: ЗАГРУЗКА ДАННЫХ С FDSN СЕРВЕРА
    # ============================================================================

    def download_event_data(self,
                            event_time: str,
                            duration_seconds: int = 3600,
                            network: str = "*",
                            station: str = "*",
                            location: str = "*",
                            channel: str = None) -> List[Dict[str, Any]]:
        """
        Загружает данные события с FDSN сервера

        Args:
            event_time: время события в формате "YYYY-MM-DDTHH:MM:SS"
            duration_seconds: длительность записи в секундах
            network: сейсмическая сеть
            station: станция
            location: локация
            channel: канал

        Returns:
            Список словарей с метаданными трасс
        """
        print(f"[1/3] Запрос данных события {event_time}...")

        t0 = UTCDateTime(event_time)
        t1 = t0 + duration_seconds

        # Загружаем waveform данные
        st = self.client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel or SACConfig.DEFAULT_CHANNELS,
            starttime=t0,
            endtime=t1,
            attach_response=True  # Важно для последующего удаления отклика
        )

        print(f"    Получено трасс: {len(st)}")
        return self._process_stream_to_sac(st, event_time)

    # ============================================================================
    # ЭТАП 2: КОНВЕРТАЦИЯ В SAC ФОРМАТ
    # ============================================================================

    def _process_stream_to_sac(self, stream, event_time: str) -> List[Dict[str, Any]]:
        """
        Конвертирует Obspy Stream в SAC файлы и извлекает метаданные

        Args:
            stream: Obspy Stream объект
            event_time: время события

        Returns:
            Список метаданных SAC файлов
        """
        print("[2/3] Конвертация в SAC формат...")

        metadata_list = []

        for tr in stream:
            try:
                # Удаляем инструментальный отклик (приводим к скорости)
                tr.remove_response(output="VEL")

                # Инициализируем SAC заголовок
                tr.stats.sac = {}

                # ==========================
                # ЗАПОЛНЕНИЕ КООРДИНАТ СТАНЦИИ
                # ==========================
                if hasattr(tr.stats, "coordinates"):
                    tr.stats.sac.stla = tr.stats.coordinates.latitude
                    tr.stats.sac.stlo = tr.stats.coordinates.longitude
                    tr.stats.sac.stel = tr.stats.coordinates.elevation
                else:
                    tr.stats.sac.stla = SACConfig.UNDEFINED_VALUE
                    tr.stats.sac.stlo = SACConfig.UNDEFINED_VALUE
                    tr.stats.sac.stel = SACConfig.UNDEFINED_VALUE

                # ==========================
                # ПАРАМЕТРЫ СОБЫТИЯ (пример для Охотского моря)
                # ==========================
                tr.stats.sac.evla = 54.9  # Широта события
                tr.stats.sac.evlo = 153.3  # Долгота события
                tr.stats.sac.evdp = 580.0  # Глубина события (км)
                tr.stats.sac.mag = 7.7  # Магнитуда

                # ==========================
                # ВРЕМЕННЫЕ МЕТКИ
                # ==========================
                tr.stats.sac.o = 0.0  # Время происхождения
                tr.stats.sac.b = 0.0  # Начало данных относительно o
                tr.stats.sac.e = tr.stats.delta * tr.stats.npts  # Конец данных

                # ==========================
                # СОХРАНЕНИЕ SAC ФАЙЛА
                # ==========================
                filename = f"{tr.id}.{tr.stats.starttime.strftime('%Y%m%dT%H%M%S')}.sac"
                filepath = self.output_dir / filename
                tr.write(str(filepath), format="SAC")

                # ==========================
                # ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ
                # ==========================
                metadata = self._extract_sac_metadata(tr, filepath)
                metadata_list.append(metadata)

                print(f"    Сохранено: {filename}")

            except Exception as e:
                print(f"    Ошибка обработки {tr.id}: {e}")

        print(f"    Всего обработано: {len(metadata_list)} файлов")
        return metadata_list

    # ============================================================================
    # ЭТАП 3: ПАРСИНГ СУЩЕСТВУЮЩИХ SAC ФАЙЛОВ
    # ============================================================================

    def parse_existing_sac_files(self, sac_dir: str) -> List[Dict[str, Any]]:
        """
        Парсит существующие SAC файлы в директории

        Args:
            sac_dir: путь к директории с SAC файлами

        Returns:
            Список метаданных SAC файлов
        """
        print(f"[3/3] Парсинг SAC файлов из {sac_dir}...")

        sac_dir = Path(sac_dir)
        metadata_list = []

        # Ищем все SAC файлы
        sac_files = list(sac_dir.glob("*.sac")) + list(sac_dir.glob("*.SAC"))

        if not sac_files:
            print("    SAC файлы не найдены!")
            return []

        print(f"    Найдено файлов: {len(sac_files)}")

        for sac_file in sac_files:
            try:
                metadata = self._parse_sac_file(sac_file)
                metadata_list.append(metadata)
                print(f"    Парсинг: {sac_file.name}")
            except Exception as e:
                print(f"    Ошибка парсинга {sac_file.name}: {e}")

        return metadata_list

    def _parse_sac_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Парсит один SAC файл
        """
        st = read(str(filepath))
        tr = st[0]

        return self._extract_sac_metadata(tr, filepath)

    def _extract_sac_metadata(self, trace, filepath: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из Obspy Trace

        Args:
            trace: Obspy Trace объект
            filepath: путь к файлу

        Returns:
            Словарь с метаданными
        """
        metadata = {
            'filename': filepath.name,
            'filepath': str(filepath),
            'network': trace.stats.network,
            'station': trace.stats.station,
            'location': trace.stats.location,
            'channel': trace.stats.channel,
            'starttime': str(trace.stats.starttime),
            'sampling_rate': trace.stats.sampling_rate,
            'npts': trace.stats.npts,
            'delta': trace.stats.delta,
        }

        # Извлекаем SAC заголовок если есть
        if hasattr(trace.stats, 'sac'):
            sac = trace.stats.sac

            # Координаты станции
            metadata.update({
                'stla': sac.stla if sac.stla != SACConfig.UNDEFINED_VALUE else None,
                'stlo': sac.stlo if sac.stlo != SACConfig.UNDEFINED_VALUE else None,
                'stel': sac.stel if sac.stel != SACConfig.UNDEFINED_VALUE else None,
            })

            # Координаты события
            metadata.update({
                'evla': sac.evla if hasattr(sac, 'evla') and sac.evla != SACConfig.UNDEFINED_VALUE else None,
                'evlo': sac.evlo if hasattr(sac, 'evlo') and sac.evlo != SACConfig.UNDEFINED_VALUE else None,
                'evdp': sac.evdp if hasattr(sac, 'evdp') else None,
                'mag': sac.mag if hasattr(sac, 'mag') else None,
            })

            # Временные параметры
            metadata.update({
                'sac_o': sac.o if hasattr(sac, 'o') else None,
                'sac_b': sac.b if hasattr(sac, 'b') else None,
                'sac_e': sac.e if hasattr(sac, 'e') else None,
            })

        # Добавляем первые N точек данных для предпросмотра
        metadata['data_preview'] = trace.data.tolist() if len(trace.data) > 0 else []

        return metadata

    # ============================================================================
    # СЕРИАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    # ============================================================================

    def _convert_numpy_types(self, obj):
        """Рекурсивно конвертирует numpy типы в стандартные Python типы"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            try:
                return float(obj)
            except:
                return str(obj)
        else:
            return obj

    def save_to_optimized_json(self,
                               metadata_list: List[Dict[str, Any]],
                               output_name: str = "sac_metadata",
                               include_data_samples: int = 10000) -> str:
        """
        Сохраняет метаданные в оптимизированный JSON файл

        Args:
            metadata_list: список метаданных
            output_name: имя выходного файла (без расширения)
            include_data_samples: сколько точек данных включать

        Returns:
            Путь к сохраненному файлу
        """
        print(f"[Сериализация] Сохранение в оптимизированный JSON...")

        # Подготовка данных для сериализации
        optimized_data = []

        for item in metadata_list:
            # Создаем копию без больших массивов
            optimized_item = {k: v for k, v in item.items()
                              if k not in ['data_preview', 'filepath']}

            # Добавляем filepath как относительный путь
            if 'filepath' in item:
                optimized_item['filepath_relative'] = os.path.relpath(
                    item['filepath'],
                    self.output_dir
                )

            # Добавляем сэмпл данных сжатый в base64
            if include_data_samples > 0 and 'data_preview' in item:
                data_array = np.array(item['data_preview'][:include_data_samples],
                                      dtype=np.float32)
                data_bytes = data_array.tobytes()
                compressed = zlib.compress(data_bytes, level=6)
                optimized_item['data_sample_b64'] = base64.b64encode(compressed).decode('ascii')

            optimized_data.append(optimized_item)

        # Структура результата
        result = {
            'metadata': optimized_data,
            'total_files': len(metadata_list),
            'output_dir': str(self.output_dir),
            'format_version': '1.0',
            'compression': 'zlib+base64'
        }

        # Сохраняем JSON
        output_path = self.output_dir / f"{output_name}.json"
        result_clean = self._convert_numpy_types(result)
        json_str = json.dumps(result_clean, separators=(',', ':'))

        # Сжимаем и сохраняем
        compressed = zlib.compress(json_str.encode('utf-8'), level=9)
        output_path_compressed = self.output_dir / f"{output_name}.json.z"

        with open(output_path_compressed, 'wb') as f:
            f.write(compressed)

        original_size = len(json_str.encode('utf-8'))
        compressed_size = len(compressed)

        print(f"    Сохранено в: {output_path_compressed}")
        print(f"    Размер: {compressed_size:,} bytes (сжатие {compressed_size / original_size * 100:.1f}%)")

        return str(output_path_compressed)

    def save_to_msgpack(self,
                        metadata_list: List[Dict[str, Any]],
                        output_name: str = "sac_metadata") -> str:
        """
        Сохраняет метаданные в MessagePack (бинарный формат)

        Args:
            metadata_list: список метаданных
            output_name: имя выходного файла

        Returns:
            Путь к сохраненному файлу
        """
        print(f"[Сериализация] Сохранение в MessagePack...")

        # Подготавливаем данные
        packed_data = {
            'files': metadata_list,
            'total': len(metadata_list),
            'timestamp': str(UTCDateTime.now())
        }

        # Сериализуем в MessagePack
        packed = msgpack.packb(packed_data, use_bin_type=True)

        # Сжимаем
        compressed = zlib.compress(packed, level=9)

        # Сохраняем
        output_path = self.output_dir / f"{output_name}.msgpack.z"
        with open(output_path, 'wb') as f:
            f.write(compressed)

        print(f"    Сохранено в: {output_path}")
        print(f"    Размер: {len(compressed):,} bytes")

        return str(output_path)


# ============================================================================
# ФУНКЦИИ ДЛЯ ВЫЗОВА ИЗ C#
# ============================================================================

def download_and_process_event(event_time: str,
                               duration: int = 3600,
                               output_dir: str = None) -> str:
    """
    Основная функция для вызова из C# - скачивает и обрабатывает данные события

    Args:
        event_time: время события "YYYY-MM-DDTHH:MM:SS"
        duration: длительность в секундах
        output_dir: папка для сохранения

    Returns:
        Путь к JSON файлу с результатами
    """
    processor = SACDataProcessor(output_dir=output_dir)

    # Скачиваем данные
    metadata = processor.download_event_data(
        event_time=event_time,
        duration_seconds=duration
    )

    # Сохраняем в оптимизированный JSON
    result_file = processor.save_to_optimized_json(metadata)

    return result_file


def parse_existing_sac_files(sac_dir: str, output_dir: str = None) -> str:
    """
    Парсит существующие SAC файлы

    Args:
        sac_dir: папка с SAC файлами
        output_dir: папка для сохранения результатов

    Returns:
        Путь к JSON файлу с результатами
    """
    processor = SACDataProcessor(output_dir=output_dir)

    # Парсим файлы
    metadata = processor.parse_existing_sac_files(sac_dir)

    # Сохраняем результаты
    if metadata:
        result_file = processor.save_to_optimized_json(metadata)
        return result_file
    else:
        return ""





# ============================================================================
# ТЕСТОВЫЙ ВЫЗОВ (если запускаем напрямую)
# ============================================================================

if __name__ == "__main__":
    # Пример 1: Скачивание данных события
    #print("=== ТЕСТ: Загрузка данных события ===")
    #result = download_and_process_event(
    #    event_time="2012-08-14T03:00:00",
    #    duration=3600,
    #    output_dir="test_output"
    #)
    #print(f"Результат: {result}")

    # Пример 2: Парсинг существующих файлов
    print("\n=== ТЕСТ: Парсинг SAC файлов ===")
    result2 = parse_existing_sac_files("C:/Users/mihas/Downloads/2012-08-14-mw77-sea-of-okhotsk")
    print(f"Результат: {result2}")