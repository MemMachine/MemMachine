# ---- builder ----
FROM node:20-alpine3.19 AS builder
RUN npm install -g mint && npm cache clean --force

# ---- runtime ----
FROM node:20-alpine3.19
WORKDIR /docs

# Copy full global prefix to preserve the mint wrapper exactly
COPY --from=builder /usr/local /usr/local

# trim common caches (optional)
RUN rm -rf /root/.npm /tmp/*

EXPOSE 3000
CMD ["mint", "dev"]
