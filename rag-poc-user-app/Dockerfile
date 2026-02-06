# Multi-stage build for React application
# Stage 1: Build the application
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Production server with serve
FROM node:18-alpine AS production

# Install serve globally
RUN npm install -g serve

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/build .

# Expose port 3000
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Start the application
CMD ["serve", "-s", ".", "-l", "3000"]
# Multi-stage build for React application
# Stage 1: Build the application
FROM node:18-alpine AS builder

ARG REACT_APP_API_URL
ARG REACT_APP_AWS_ACCESS_KEY_ID
ARG REACT_APP_AWS_SECRET_ACCESS_KEY
ARG REACT_APP_AWS_REGION

ENV REACT_APP_API_URL=${REACT_APP_API_URL}
ENV REACT_APP_AWS_ACCESS_KEY_ID=${REACT_APP_AWS_ACCESS_KEY_ID}
ENV REACT_APP_AWS_SECRET_ACCESS_KEY=${REACT_APP_AWS_SECRET_ACCESS_KEY}
ENV REACT_APP_AWS_REGION=${REACT_APP_AWS_REGION}

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Production server with serve
FROM node:18-alpine AS production

# Install serve globally
RUN npm install -g serve

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/build .

# Expose port 3000
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Start the application
CMD ["serve", "-s", ".", "-l", "3000"]
