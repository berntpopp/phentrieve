FROM node:lts-alpine as build-stage

WORKDIR /app

# Copy package files first for better caching
COPY package.json package-lock.json* ./

# Install build dependencies
RUN apk add --no-cache --virtual .build-deps \
    autoconf \
    automake \
    build-base \
    libtool \
    nasm \
    zlib-dev \
    libpng-dev \
    libjpeg-turbo-dev \
    libwebp-dev \
    pkgconfig \
    python3

# Install dependencies with clean environment
RUN npm ci --no-audit --no-fund

# Clean up build dependencies
RUN apk del .build-deps

# Copy project files excluding node_modules (should be in .dockerignore)
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM fholzer/nginx-brotli:v1.28.0 as production-stage

# Copy built files and config
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# Remove default nginx config to avoid conflicts
RUN rm -f /etc/nginx/conf.d/*.conf.default

EXPOSE 80

# Start nginx in foreground mode
ENTRYPOINT ["nginx"]
CMD ["-g", "daemon off;"]
