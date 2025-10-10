# cv_factory/shared_libs/data_ingestion/connectors/kafka_connector.py

import logging
import json
import time
from typing import Dict, Any, Iterator, Optional, Union
from confluent_kafka import Consumer, Producer, KafkaException, OFFSET_BEGINNING

# Import Base Abstraction
from ..base.base_stream_connector import BaseStreamConnector, StreamData 
# BaseStreamConnector enforces connect, consume, produce, close

logger = logging.getLogger(__name__)

class KafkaConnector(BaseStreamConnector):
    """
    A concrete stream connector for Apache Kafka. 
    
    Acts as both a Consumer (for model input) and a Producer (for inference output), 
    adhering to the BaseStreamConnector contract.
    """

    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the KafkaConnector, but does not establish connections yet.

        Args:
            connector_id (str): Unique ID.
            config (Optional[Dict[str, Any]]): Configuration dictionary including 
                                               'consumer_config' and 'producer_config'.
        """
        super().__init__(connector_id, config)
        self.consumer: Optional[Consumer] = None
        self.producer: Optional[Producer] = None
        self.consumer_config = self.config.get('consumer_config', {})
        self.producer_config = self.config.get('producer_config', {})
        self.input_topic = self.config.get('input_topic')
        
    def connect(self) -> bool:
        """
        Establishes connections for both Kafka Consumer and Producer.
        """
        if self.is_connected:
            return True
            
        try:
            # 1. Initialize Consumer
            if self.input_topic:
                if not self.consumer_config.get('group.id'):
                    # Production best practice: always use a group.id
                    self.consumer_config['group.id'] = f"{self.connector_id}_group"
                    
                self.consumer_config['auto.offset.reset'] = self.consumer_config.get('auto.offset.reset', 'earliest')
                self.consumer = Consumer(self.consumer_config)
                self.consumer.subscribe([self.input_topic])
                logger.info(f"[{self.connector_id}] Kafka Consumer initialized for topic: {self.input_topic}")
            
            # 2. Initialize Producer
            self.producer = Producer(self.producer_config)
            logger.info(f"[{self.connector_id}] Kafka Producer initialized.")
            
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to connect to Kafka: {e}")
            self.is_connected = False
            raise ConnectionError(f"Kafka connection failed: {e}")

    def consume(self, timeout_ms: int = 1000, max_batches: Optional[int] = None, **kwargs) -> Iterator[StreamData]:
        """
        Consumes messages from the input topic and yields them.

        Args:
            timeout_ms (int): Max time to wait for a message.
            max_batches (Optional[int]): Max number of messages to yield before stopping.
        
        Yields:
            StreamData: Decoded message data (assumed JSON or text).
        """
        if not self.consumer:
            raise RuntimeError("Consumer is not initialized. Check input_topic config.")
        
        message_count = 0
        while self.is_connected and (max_batches is None or message_count < max_batches):
            try:
                # Poll for message
                msg = self.consumer.poll(timeout=timeout_ms / 1000.0)
                
                if msg is None:
                    # No message, continue polling
                    continue
                
                if msg.error():
                    if msg.error().fatal():
                        raise KafkaException(msg.error())
                    logger.warning(f"Kafka Consumer error: {msg.error()}")
                    continue

                # Decode the message value (assuming UTF-8 JSON encoding)
                data = msg.value().decode('utf-8')
                
                # Try to parse JSON (common for production streams)
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    yield data # Yield as raw string if not JSON

                message_count += 1

            except KafkaException as e:
                logger.error(f"Kafka exception during consumption: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error during stream consumption: {e}")
                break

    def produce(self, data: StreamData, destination_topic: str, key: Optional[str] = None, **kwargs):
        """
        Writes/Produces data (e.g., inference results) to a destination Kafka topic.

        Args:
            data (StreamData): The data payload (will be serialized to JSON).
            destination_topic (str): The Kafka topic to write to.
            key (Optional[str]): Message key for partitioning.
        """
        if not self.producer:
            raise RuntimeError("Producer is not initialized.")
            
        # Serialize payload to JSON bytes
        payload = json.dumps(data).encode('utf-8')
        
        # Delivery Report Callback (for asynchronous tracking)
        def delivery_report(err, msg):
            if err is not None:
                logger.error(f"[{self.connector_id}] Message delivery failed to {msg.topic()}: {err}")
            else:
                logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

        try:
            self.producer.produce(
                topic=destination_topic, 
                value=payload, 
                key=key, 
                callback=delivery_report
            )
            
            # Flush periodically to ensure delivery in high-throughput scenarios
            if kwargs.get('flush_sync', False):
                 self.producer.flush()
            
        except BufferError:
            # Handle full buffer: poll to process delivery reports and retry
            self.producer.poll(0)
            self.producer.produce(topic=destination_topic, value=payload, key=key, callback=delivery_report)
        except Exception as e:
            logger.error(f"Failed to produce message to {destination_topic}: {e}")
            raise

    def close(self):
        """
        Safely closes the Kafka Consumer and Producer connections.
        """
        if self.consumer:
            self.consumer.close()
            logger.info(f"[{self.connector_id}] Kafka Consumer closed.")
        
        if self.producer:
            # Ensure all buffered messages are delivered before closing
            self.producer.flush()
            logger.info(f"[{self.connector_id}] Kafka Producer flushed and closed.")
            
        super().close()