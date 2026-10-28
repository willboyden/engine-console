//! The HTTP API layer: routing, validation, security and the route handlers (each calls one service method).
pub mod deps;
pub mod http;
pub mod router;
pub mod routes;
pub mod schemas;
pub mod security;
pub mod statics;
pub mod validate;
