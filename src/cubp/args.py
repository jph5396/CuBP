import argparse
import os
import sys
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Backend(StrEnum):
    numpy = "numpy"
    cupy = "cupy"
    cuda = "cuda"


class ImageBounds(BaseModel):
    x: int = 512
    y: int = 512


class Target(BaseModel):
    lon: float | None = None
    lat: float | None = None
    alt: float | None = None

    @model_validator(mode="after")
    def all_or_nothing(self) -> "Target":
        fields = [self.lon, self.lat, self.alt]
        if any(f is not None for f in fields) and not all(f is not None for f in fields):
            raise ValueError("target requires lat, lon, and alt to all be set")
        return self


class CuBPArguments(BaseSettings):
    config: Path | None = Field(default=None, description="Path to YAML config file.")
    cphd_file: Path = Field(description="Path to CPHD file to run bp on.")
    image_bounds: ImageBounds = Field(default=ImageBounds(), description="Image bounds for produced image.")
    pulse_limit: int = Field(default=-1, description="number of pulses to use during image formation.")
    image_spacing: float = Field(default=0.5, description="spacing between pixels in meters.")
    output_file: Path = Field(default=Path("out.png"), description="path to output a formed image to.")
    target: Target | None = Field(default=None, description="geodetic location of target")
    backend: Backend = Field(default=Backend.numpy, description="Backend implementation to use.")
    log_file: str | None = Field(default=None, description="optional file to write logs to.")
    log_level: str = Field(default="DEBUG", description="log level to be used.")
    concurrency_limit: int = Field(
        default=1,
        description="Concurrency limit for a given run. This will also increase the amount of memory used.",
    )

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="CUBP_",
        env_nested_delimiter="__",
    )

    @field_validator("cphd_file")
    @classmethod
    def cphd_file_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"CPHD file not found: {v}")
        return v

    @field_validator("config")
    @classmethod
    def config_file_must_exist(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"config file not found: {v}")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: BaseSettings,
        **kwargs,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Pre-parse just the --config argument so we can load the YAML source
        # before the full settings parse runs. parse_known_args ignores all
        # other arguments so it won't conflict with CliSettingsSource.
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--config", default=os.getenv("CUBP_CONFIG"))
        pre_args, _ = pre_parser.parse_known_args(sys.argv[1:])

        sources: list[PydanticBaseSettingsSource] = [
            CliSettingsSource(settings_cls, cli_parse_args=True),  # pyright: ignore
        ]

        if pre_args.config:
            sources.append(
                YamlConfigSettingsSource(settings_cls, yaml_file=pre_args.config)  # pyright: ignore[reportArgumentType]
            )

        return tuple(sources)
