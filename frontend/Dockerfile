FROM node:lts-alpine as build-stage

WORKDIR /app

COPY package*.json ./

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

RUN npm install

# Clean up build dependencies
RUN apk del .build-deps

COPY . .
RUN npm run build

# Production stage
FROM fholzer/nginx-brotli:v1.28.0 as production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
