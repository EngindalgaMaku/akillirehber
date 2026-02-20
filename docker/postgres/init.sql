-- PostgreSQL initialization script for RAG Educational Chatbot
-- This script runs when the container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Grant privileges to the application user
GRANT ALL PRIVILEGES ON DATABASE ragchatbot TO raguser;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
END $$;
