#!/usr/bin/env python3
"""
Qdrant Data Migration Script
This script helps migrate Qdrant data between different environments.
"""

import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantMigrator:
    def __init__(self, source_url: str, target_url: str, source_api_key: str = None, target_api_key: str = None):
        self.source_client = QdrantClient(url=source_url, api_key=source_api_key)
        self.target_client = QdrantClient(url=target_url, api_key=target_api_key)
        
    async def export_collection(self, collection_name: str, output_file: str):
        """Export a collection to JSON file."""
        logger.info(f"Exporting collection '{collection_name}' to {output_file}")
        
        try:
            # Get collection info
            collection_info = self.source_client.get_collection(collection_name)
            logger.info(f"Collection info: {collection_info}")
            
            # Get all points
            points = []
            offset = None
            
            while True:
                # Retrieve points in batches
                result = self.source_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points_batch = result[0]
                if not points_batch:
                    break
                    
                points.extend(points_batch)
                offset = result[1]
                
                logger.info(f"Retrieved {len(points_batch)} points (total: {len(points)})")
            
            # Prepare export data
            export_data = {
                "collection_name": collection_name,
                "collection_info": {
                    "vectors": collection_info.config.params.vectors,
                    "optimizer_config": collection_info.config.params.optimizer_config,
                    "hnsw_config": collection_info.config.params.hnsw_config,
                    "wal_config": collection_info.config.params.wal_config,
                    "quantization_config": collection_info.config.params.quantization_config
                },
                "points": [
                    {
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload
                    }
                    for point in points
                ],
                "total_points": len(points)
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Successfully exported {len(points)} points to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting collection: {str(e)}")
            return False
    
    async def import_collection(self, input_file: str, collection_name: str = None):
        """Import a collection from JSON file."""
        logger.info(f"Importing collection from {input_file}")
        
        try:
            with open(input_file, 'r') as f:
                export_data = json.load(f)
            
            # Use provided collection name or the one from the file
            target_collection = collection_name or export_data["collection_name"]
            
            # Create collection if it doesn't exist
            try:
                self.target_client.get_collection(target_collection)
                logger.info(f"Collection '{target_collection}' already exists")
            except:
                logger.info(f"Creating collection '{target_collection}'")
                
                # Extract vector configuration
                vectors_config = export_data["collection_info"]["vectors"]
                if isinstance(vectors_config, dict):
                    # Multiple vectors
                    vector_params = {
                        name: VectorParams(
                            size=config["size"],
                            distance=Distance(config["distance"])
                        )
                        for name, config in vectors_config.items()
                    }
                else:
                    # Single vector
                    vector_params = VectorParams(
                        size=vectors_config["size"],
                        distance=Distance(vectors_config["distance"])
                    )
                
                self.target_client.create_collection(
                    collection_name=target_collection,
                    vectors_config=vector_params,
                    optimizers_config=export_data["collection_info"].get("optimizer_config"),
                    hnsw_config=export_data["collection_info"].get("hnsw_config"),
                    wal_config=export_data["collection_info"].get("wal_config"),
                    quantization_config=export_data["collection_info"].get("quantization_config")
                )
            
            # Import points in batches
            points = export_data["points"]
            batch_size = 100
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                point_structs = [
                    PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]
                    )
                    for point in batch
                ]
                
                self.target_client.upsert(
                    collection_name=target_collection,
                    points=point_structs
                )
                
                logger.info(f"Imported batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully imported {len(points)} points to '{target_collection}'")
            return True
            
        except Exception as e:
            logger.error(f"Error importing collection: {str(e)}")
            return False
    
    async def list_collections(self, client_name: str = "source"):
        """List all collections."""
        client = self.source_client if client_name == "source" else self.target_client
        
        try:
            collections = client.get_collections()
            logger.info(f"{client_name.title()} collections:")
            for collection in collections.collections:
                info = client.get_collection(collection.name)
                logger.info(f"  - {collection.name}: {info.points_count} points")
            return collections.collections
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

async def main():
    parser = argparse.ArgumentParser(description="Qdrant Data Migration Tool")
    parser.add_argument("--source-url", help="Source Qdrant URL")
    parser.add_argument("--target-url", help="Target Qdrant URL")
    parser.add_argument("--source-api-key", help="Source Qdrant API key")
    parser.add_argument("--target-api-key", help="Target Qdrant API key")
    parser.add_argument("--action", choices=["export", "import", "list"], required=True, help="Action to perform")
    parser.add_argument("--collection", help="Collection name")
    parser.add_argument("--file", help="Export/import file path")
    parser.add_argument("--client", choices=["source", "target"], default="source", help="Client for list action")
    
    args = parser.parse_args()
    
    # Set default URLs if not provided
    if not args.source_url:
        args.source_url = "http://localhost:6333"
    if not args.target_url:
        args.target_url = "http://localhost:6333"
    
    migrator = QdrantMigrator(
        source_url=args.source_url,
        target_url=args.target_url,
        source_api_key=args.source_api_key,
        target_api_key=args.target_api_key
    )
    
    if args.action == "export":
        if not args.collection or not args.file:
            logger.error("Collection name and file path are required for export")
            return
        await migrator.export_collection(args.collection, args.file)
    
    elif args.action == "import":
        if not args.file:
            logger.error("File path is required for import")
            return
        await migrator.import_collection(args.file, args.collection)
    
    elif args.action == "list":
        await migrator.list_collections(args.client)

if __name__ == "__main__":
    asyncio.run(main())
