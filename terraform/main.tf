provider "aws" {
  region = "eu-west-1"  # Set your desired AWS region
}

terraform {
  backend "s3" {
    bucket         = "openclassrooms-tfstates-xavier"
    key            = "project7/terraform.tfstate"
    region         = "eu-west-1"  # Set the region where your S3 bucket is located
    encrypt        = true
  }
}

resource "aws_instance" "Project-7" {
  ami           = "ami-0e309a5f3a6dd97ea"  # Specify the AMI ID for the desired Amazon Machine Image
  instance_type = "t2.micro"  # Specify the instance type

  tags = {
    Name = "Project-7"
  }
}

