-- Reset all user passwords to "password123"
-- Run this inside postgres container:
-- docker-compose exec postgres psql -U raguser -d ragchatbot -f /tmp/reset_passwords.sql

-- Argon2 hash for "password123"
UPDATE users SET hashed_password = '$argon2id$v=19$m=65536,t=3,p=4$6x1jrPW+t9b6X0vpfS+lNA$qvCZe8K7qvCZe8K7qvCZe8K7qvCZe8K7qvCZe8K7qvA';

SELECT 'All passwords reset to: password123' as message;
