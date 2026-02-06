"""
Basic usage example for the Advanced RAG System.
This script demonstrates how to process documents and query the system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.rag_pipeline import create_rag_pipeline


async def main():
    """Main example function demonstrating RAG system usage."""
    
    print("🚀 Advanced RAG System - Basic Usage Example")
    print("=" * 50)
    
    try:
        # Initialize the RAG pipeline
        print("1. Initializing RAG pipeline...")
        pipeline = await create_rag_pipeline()
        print("✅ Pipeline initialized successfully!")
        
        # Check system status
        print("\n2. Checking system status...")
        stats = await pipeline.get_system_stats()
        print(f"✅ Vector store status: {stats.get('vector_store', {}).get('status', 'Unknown')}")
        print(f"✅ Total documents: {stats.get('total_documents', 0)}")
        
        # Example 1: Process documents (if available)
        print("\n3. Document Processing Example")
        print("-" * 30)
        
        # Check if there are example documents in the examples directory
        examples_dir = Path(__file__).parent
        pdf_files = list(examples_dir.glob("*.pdf"))
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files to process:")
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name}")
            
            # Process the documents
            print("\nProcessing documents...")
            file_paths = [str(pdf_file) for pdf_file in pdf_files]
            result = await pipeline.process_documents(file_paths)
            
            print(f"✅ Processing completed:")
            print(f"  - Total files: {result['total_files']}")
            print(f"  - Successful: {result['successful_files']}")
            print(f"  - Failed: {len(result['failed_files'])}")
            print(f"  - Total chunks created: {result['total_chunks']}")
            print(f"  - Processing time: {result['processing_time']:.2f} seconds")
            
            if result['failed_files']:
                print("❌ Failed files:")
                for failed_file in result['failed_files']:
                    print(f"  - {failed_file}")
        else:
            print("ℹ️  No PDF files found in examples directory.")
            print("   Add some PDF files to test document processing.")
        
        # Example 2: Query the system
        print("\n4. Querying Example")
        print("-" * 20)
        
        # List of example queries
        example_queries = [
            "What is the main purpose of this document?",
            "How do I get started?",
            "What are the key features?",
            "Can you summarize the installation process?",
            "What are the system requirements?"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * (len(query) + 10))
            
            # Execute the query
            result = await pipeline.query(
                query=query,
                use_query_expansion=True,
                use_reranking=True
            )
            
            print(f"Status: {result.status}")
            print(f"Confidence: {result.confidence_score:.3f}")
            print(f"Answer: {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")
            
            if result.sources:
                print(f"Sources ({len(result.sources)}):")
                for j, source in enumerate(result.sources[:3], 1):  # Show top 3 sources
                    print(f"  {j}. {source['document_name']} (Page {source['page_number']}) - Score: {source['relevance_score']:.3f}")
            
            print(f"Processing time: {result.processing_time:.2f}s")
            
            # Only run first query if no documents are available
            if stats.get('total_documents', 0) == 0:
                print("\nℹ️  Skipping remaining queries - no documents available.")
                break
        
        # Example 3: Advanced query with filters
        if stats.get('total_documents', 0) > 0:
            print("\n5. Advanced Query with Filters")
            print("-" * 35)
            
            # Get list of available documents
            documents = stats.get('document_list', [])
            if documents:
                first_doc = documents[0]
                print(f"Filtering by document: {first_doc}")
                
                result = await pipeline.query(
                    query="What information is available in this document?",
                    use_query_expansion=False,  # Disable for focused search
                    use_reranking=True,
                    document_filter=first_doc
                )
                
                print(f"✅ Filtered query result:")
                print(f"  Status: {result.status}")
                print(f"  Confidence: {result.confidence_score:.3f}")
                print(f"  Answer: {result.answer[:150]}{'...' if len(result.answer) > 150 else ''}")
        
        # Example 4: System statistics
        print("\n6. Final System Statistics")
        print("-" * 28)
        
        final_stats = await pipeline.get_system_stats()
        vector_info = final_stats.get('vector_store', {})
        
        print(f"✅ Collection status: {vector_info.get('status', 'Unknown')}")
        print(f"✅ Total points: {vector_info.get('points_count', 0)}")
        print(f"✅ Vector dimension: {vector_info.get('vector_size', 0)}")
        print(f"✅ Distance metric: {vector_info.get('distance', 'Unknown')}")
        print(f"✅ Total documents: {final_stats.get('total_documents', 0)}")
        
        config = final_stats.get('pipeline_config', {})
        print(f"\n📋 Pipeline Configuration:")
        print(f"  - Model: {config.get('model', 'Unknown')}")
        print(f"  - Embedding model: {config.get('embedding_model', 'Unknown')}")
        print(f"  - Max sources: {config.get('max_sources', 0)}")
        print(f"  - Confidence threshold: {config.get('min_confidence_threshold', 0)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all API keys are set in your .env file")
        print("2. Make sure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)")
        print("3. Check that you have sufficient API quotas")
        print("4. Verify your internet connection")
        return 1
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Add your own PDF documents to the examples/ directory")
    print("2. Try the FastAPI server: python -m api.main")
    print("3. Explore the API documentation at http://localhost:8000/docs")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 