# API Reference

## Overview

This document describes the API endpoints for our service.

## Authentication

All API requests require authentication using Bearer tokens.

```
Authorization: Bearer <token>
```

## Endpoints

### GET /api/users

Returns a list of users.

**Response:**

```json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"
    }
  ]
}
```

### POST /api/users

Create a new user.

**Request Body:**

```json
{
  "name": "Jane Doe",
  "email": "jane@example.com"
}
```

**Response:**

```json
{
  "id": 2,
  "name": "Jane Doe",
  "email": "jane@example.com",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### GET /api/documents/{id}

Retrieve a specific document by ID.

**Response:**

```json
{
  "id": "doc_123",
  "title": "Document Title",
  "content": "Document content...",
  "author": "John Doe"
}
```

### DELETE /api/documents/{id}

Delete a document.

**Response:** 204 No Content

## Error Handling

All errors return a consistent format:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found"
  }
}
```

## Rate Limiting

API requests are limited to 100 requests per minute per API key.

## Versioning

API versioning is done via URL path: `/api/v1/`, `/api/v2/`, etc.