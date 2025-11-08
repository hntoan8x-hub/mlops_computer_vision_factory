# cv_factory/shared_libs/data_ingestion/connectors/kafka_connector.py

import logging
import json
import time
from typing import Dict, Any, Iterator, Optional, Union
from confluent_kafka import Consumer, Producer, KafkaException, OFFSET_BEGINNING, Message

# Import Base Abstraction
from ..base.base_stream_connector import BaseStreamConnector, StreamData 
# BaseStreamConnector enforces connect, consume, produce, close

logger = logging.getLogger(__name__)

class KafkaConnector(BaseStreamConnector):
    """
    A concrete stream connector for Apache Kafka. 
    
    Acts as both a Consumer (for model input) and a Producer (for inference output), 
    adhering to the BaseStreamConnector contract.

    Attributes:
        consumer: The confluent_kafka Consumer instance.
        producer: The confluent_kafka Producer instance.
        consumer_config: Configuration dictionary for the consumer.
        producer_config: Configuration dictionary for the producer.
        input_topic: The topic the connector consumes from.
    """

    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the KafkaConnector, but does not establish connections yet.

        Args:
            connector_id: Unique ID.
            config: Configuration dictionary including 'consumer_config' and 'producer_config'.
        """
        super().__init__(connector_id, config)
        self.consumer: Optional[Consumer] = None
        self.producer: Optional[Producer] = None
        # Default to empty dicts for safety
        self.consumer_config = self.config.get('consumer_config', {})
        self.producer_config = self.config.get('producer_config', {})
        self.input_topic = self.config.get('input_topic')
        
    def connect(self) -> bool:
        """
        Establishes connections for both Kafka Consumer and Producer.

        Returns:
            bool: True if connections are successful.

        Raises:
            ConnectionError: If connection to Kafka brokers fails.
        """
        if self.is_connected:
            return True
            
        try:
            # 1. Initialize Consumer
            if self.input_topic:
                if not self.consumer_config.get('group.id'):
                    # Hardening: Always use a meaningful group.id
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
        Consumes messages from the input topic and yields decoded data.

        Args:
            timeout_ms: Max time (in milliseconds) to wait for a message.
            max_batches: Max number of messages to yield before stopping (optional).
            **kwargs: Additional consumer configuration.
        
        Yields:
            StreamData: Decoded message data (assumed JSON or raw string).

        Raises:
            RuntimeError: If consumer is not initialized.
        """
        if not self.consumer:
            raise RuntimeError("Consumer is not initialized. Check input_topic config.")
        
        message_count = 0
        while self.is_connected and (max_batches is None or message_count < max_batches):
            try:
                # Poll for message
                msg: Message = self.consumer.poll(timeout=timeout_ms / 1000.0)
                
                if msg is None:
                    # No message, continue polling (handle stream idleness gracefully)
                    continue
                
                if msg.error():
                    if msg.error().fatal():
                        # Fatal error (e.g., broker disconnect), stop consuming
                        logger.critical(f"Fatal Kafka Consumer error: {msg.error()}")
                        self.is_connected = False 
                        break
                    
                    # Non-fatal errors (e.g., temporary network issue), log and continue
                    logger.warning(f"Kafka Consumer non-fatal error: {msg.error()}")
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
                # Catch unexpected Kafka errors during the polling loop
                logger.error(f"Kafka exception during consumption: {e}")
                time.sleep(1) # Wait briefly before retrying poll
                continue
            except Exception as e:
                logger.error(f"Unexpected error during stream consumption: {e}")
                self.is_connected = False
                break

    def produce(self, data: StreamData, destination_topic: str, key: Optional[str] = None, **kwargs):
        """
        Writes/Produces data (e.g., inference results) to a destination Kafka topic.

        Args:
            data: The data payload (will be serialized to JSON).
            destination_topic: The Kafka topic to write to.
            key: Message key for partitioning (optional).
            **kwargs: Optional parameters, including 'flush_sync' (bool).

        Raises:
            RuntimeError: If producer is not initialized or synchronous flush fails.
        """
        if not self.producer:
            raise RuntimeError("Producer is not initialized.")
            
        # Serialize payload to JSON bytes
        payload = json.dumps(data).encode('utf-8')
        
        # Delivery Report Callback (for asynchronous tracking and error handling)
        def delivery_report(err, msg):
            """Called once for each message produced to indicate delivery result."""
            if err is not None:
                # Critical: Log failed delivery for audit/reconciliation
                logger.error(f"[{self.connector_id}] Message delivery failed to {msg.topic()} (Key: {msg.key()}): {err}")
            else:
                logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

        try:
            self.producer.produce(
                topic=destination_topic, 
                value=payload, 
                key=key, 
                callback=delivery_report
            )
            
            # Hardening: Optional synchronous flush check for critical data
            if kwargs.get('flush_sync', False):
                 # Use a reasonable timeout (e.g., 5 seconds)
                 remaining_messages = self.producer.flush(timeout=5.0) 
                 if remaining_messages > 0:
                     logger.error(f"[{self.connector_id}] Failed to deliver {remaining_messages} messages during synchronous flush.")
                     # Raise error to halt the production process if sync is critical
                     raise RuntimeError(f"Kafka producer timeout on synchronous flush.")
            
        except BufferError:
            # Handle full buffer: poll to process delivery reports and retry
            logger.warning(f"[{self.connector_id}] Producer buffer full, polling and retrying.")
            self.producer.poll(0)
            self.producer.produce(topic=destination_topic, value=payload, key=key, callback=delivery_report)
        except Exception as e:
            logger.error(f"Failed to produce message to {destination_topic}: {e}")
            raise

    def close(self):
        """
        Safely closes the Kafka Consumer and Producer connections.
        
        Ensures all buffered producer messages are delivered using a timeout.
        """
        if self.consumer:
            self.consumer.close()
            logger.info(f"[{self.connector_id}] Kafka Consumer closed.")
        
        if self.producer:
            # Hardening: Use a graceful timeout (e.g., 10 seconds) when flushing remaining messages
            # This is CRITICAL to prevent data loss on shutdown.
            timeout_s = 10.0
            remaining = self.producer.flush(timeout=timeout_s) 
            if remaining > 0:
                logger.error(f"[{self.connector_id}] WARNING: {remaining} messages unsent after {timeout_s}s flush timeout. Potential data loss.")
            else:
                 logger.info(f"[{self.connector_id}] Kafka Producer flushed all messages successfully.")
            
        super().close()