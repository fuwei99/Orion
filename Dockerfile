FROM golang:1.23-alpine AS builder
WORKDIR /src

COPY go.mod ./
COPY main.go ./

RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/onyx2api .

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /

COPY --from=builder /out/onyx2api /onyx2api

ENV PORT=7860
EXPOSE 7860

ENTRYPOINT ["/onyx2api"]
