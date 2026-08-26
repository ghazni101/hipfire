FROM local/rocm-base:7.14.0

COPY hipfire daemon /opt/hipfire/bin/
COPY config.toml /opt/hipfire/home/config.toml
COPY registry.cache.json /opt/hipfire/home/registry.cache.json
COPY serve.sh /opt/hipfire/serve.sh
RUN chmod +x /opt/hipfire/serve.sh /opt/hipfire/bin/hipfire /opt/hipfire/bin/daemon

ENV ROCM_PATH=/opt/rocm \
    HIPFIRE_HOME=/opt/hipfire/home \
    HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
    PATH=/opt/hipfire/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ORNITH_KV_MODE=fwht4 \
    HIPFIRE_REGISTRY_URL=http://127.0.0.1:9/v1.json

EXPOSE 8420
ENTRYPOINT ["/opt/hipfire/serve.sh"]
