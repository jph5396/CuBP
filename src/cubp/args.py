from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    CliSettingsSource
)

from pydantic import BaseModel, Field
from enum import StrEnum
from pathlib import Path
import os

class Backend(StrEnum): 
    numpy = "numpy"
    cuda = "cuda"

class ImageBounds(BaseModel): 
    x: int = 512
    y: int = 512

class TargetLonLat(BaseModel): 
    lon: float | None = None
    lat: float | None = None



class CuBPArguments(BaseSettings):
    cphd_file: str = Field(default = "test", description = "Path to CPHD file to run bp on.")
    image_bounds: ImageBounds = Field(default= ImageBounds(), description = "Image bounds for produced image.")
    pulse_limit: int = Field(default= -1, description= "number of pulses to use during image formation.")
    image_spacing: float = Field(default = .5, description="spacing between pixels in meters.")
    output_file: Path = Field(default = Path('out.png'), description = "path to output a formed image to.")
    target_lon_lat: TargetLonLat | None = Field( default = None, description= " geodetic location of target")
    backend: Backend = Field(default = Backend.numpy, description="Backend implementation to use.")


    model_config = SettingsConfigDict(
        cli_parse_args=True, 
        yaml_file= os.getenv("CuBP_CONFIG")
    )

    @classmethod
    def settings_customise_sources(
        cls, 
        settings_cls: BaseSettings, 
        **kwargs, 
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        return (
            CliSettingsSource(settings_cls, cli_parse_args= True), # pyright: ignore[reportArgumentType]
            YamlConfigSettingsSource(settings_cls) # pyright: ignore[reportArgumentType]
        )

