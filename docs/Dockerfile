FROM alpine:latest

# Install Node.js and npm
RUN apk add --no-cache nodejs npm

# Install Mintlify globally
RUN npm install -g mint

# Create working directory
WORKDIR /docs

# Expose port (Mintlify dev server default)
EXPOSE 3000

# Default command
CMD ["mint", "dev"]

