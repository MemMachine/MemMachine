FROM node:22-alpine AS builder

# Install Mintlify globally
RUN npm install -g mint && \
    npm cache clean --force && \
    rm -rf /tmp/* /root/.npm

FROM node:22-alpine

# Copy only the global Mintlify installation from builder
COPY --from=builder /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=builder /usr/local/bin/mint /usr/local/bin/mint

# Create working directory
WORKDIR /docs

# Expose port (Mintlify dev server default)
EXPOSE 3000

# Default command
CMD ["mint", "dev"]
